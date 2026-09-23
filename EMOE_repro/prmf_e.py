"""Minimal E-question adapter around pinned P-RMF core modules.

Keeps the author's modality token encoders, three VAEs, proxy uncertainty
fusion, gradient reversal, shared-layer crossmodal injection and reconstructor.
Only replaces the BERT entry with aligned_50 continuous Text and adds a 3-way
classification head alongside the original scalar regression head.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from P_RMF_upstream.models.basic_layers import Transformer, CrossmodalEncoder, GradientReversalLayer
from P_RMF_upstream.models.generate_proxy_modality import Generate_Proxy_Modality


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "P_RMF_upstream" / "configs" / "train_mosei.yaml"


def prmf_config():
    with CONFIG.open(encoding="utf-8") as stream:
        args = deepcopy(yaml.safe_load(stream))
    feature = args["model"]["feature_extractor"]
    feature["input_length"] = [50, 50, 50]
    feature["input_dims"] = [768, 35, 74]
    feature["bert_pretrained"] = None
    args["dataset"]["dataPath"] = "attachment2/aligned_50.pkl"
    return args


class PRMFE(nn.Module):
    """Author's P-RMF architecture, with E Q2 input and dual-task outputs."""

    def __init__(self, cfg=None, *, missing_embedding=False, coverage_gate=False):
        super().__init__()
        cfg = prmf_config() if cfg is None else cfg
        self.cfg = cfg
        self.use_missing_embedding = missing_embedding
        self.use_coverage_gate = coverage_gate
        feature = cfg["model"]["feature_extractor"]

        def projection(index):
            return nn.Sequential(
                nn.Linear(feature["input_dims"][index], feature["hidden_dims"][index]),
                Transformer(num_frames=feature["input_length"][index], save_hidden=False,
                            token_len=feature["token_length"][index],
                            dim=feature["hidden_dims"][index], depth=feature["depth"],
                            heads=feature["heads"], mlp_dim=feature["hidden_dims"][index]))

        self.proj_l = projection(0)
        self.proj_v = projection(1)
        self.proj_a = projection(2)
        proxy = cfg["model"]["generate_proxy"]
        self.generate_proxy_modality = Generate_Proxy_Modality(
            cfg, proxy["input_dim"], proxy["hidden_dim"], proxy["out_dim"])
        self.GRL = GradientReversalLayer(alpha=1.0)
        reconstruction = cfg["model"]["reconstructor"]
        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=reconstruction["input_length"], save_hidden=False,
                        token_len=None, dim=reconstruction["input_dim"],
                        depth=reconstruction["depth"], heads=reconstruction["heads"],
                        mlp_dim=reconstruction["hidden_dim"])
            for _ in range(3)])
        cross = cfg["model"]["crossmodal_encoder"]
        self.crossmodal_encoder = CrossmodalEncoder(
            proxy_dim=cross["proxy_dim"], text_dim=cross["hidden_dims"][0],
            audio_dim=cross["hidden_dims"][2], video_dim=cross["hidden_dims"][1],
            embed_dim=cross["embed_dim"], num_layers=cross["num_layers"],
            attn_dropout=cross["attn_dropout"])
        regression = cfg["model"]["regression"]
        self.fc1 = nn.Linear(regression["input_dim"], regression["hidden_dim"])
        self.fc2 = nn.Linear(regression["hidden_dim"], 1)
        self.dropout = nn.Dropout(regression["attn_dropout"])
        self.cls_head = nn.Linear(regression["input_dim"], 3)
        # E-question variants only. Zero initialization preserves the baseline
        # at step zero and consumes no RNG for the shared author-core layers.
        if missing_embedding:
            self.missing_embedding = nn.Parameter(torch.zeros(3, feature["hidden_dims"][0]))
        if coverage_gate:
            self.coverage_beta = nn.Parameter(torch.zeros(()))

    def forward(self, text, audio, vision, observed_mask, complete_for_aux=False,
                valid_lengths=None):
        if text.ndim != 3 or tuple(text.shape[1:]) != (50, 768):
            raise ValueError("Text must be B×50×768 aligned continuous features")
        if audio.shape != (len(text), 50, 74) or vision.shape != (len(text), 50, 35):
            raise ValueError("Audio/Vision must be B×50×74 and B×50×35")
        if observed_mask.shape != (len(text), 3, 50):
            raise ValueError("observed_mask must be B×3×50, L/V/A order")
        # Original P-RMF represents removed features by zero; unlike EMOE it
        # has no explicit missing token. The same input mask is shared by both.
        incomplete_l = text * observed_mask[:, 0, :, None]
        incomplete_v = vision * observed_mask[:, 1, :, None]
        incomplete_a = audio * observed_mask[:, 2, :, None]
        if self.use_missing_embedding:
            def encode(proj, values, mask, channel):
                projected = proj[0](values)
                projected = projected + (~mask).to(projected.dtype)[..., None] * self.missing_embedding[channel]
                return proj[1](projected)[:, :8]

            h_l = encode(self.proj_l, incomplete_l, observed_mask[:, 0], 0)
            h_v = encode(self.proj_v, incomplete_v, observed_mask[:, 1], 1)
            h_a = encode(self.proj_a, incomplete_a, observed_mask[:, 2], 2)
        else:
            h_l = self.proj_l(incomplete_l)[:, :8]
            h_v = self.proj_v(incomplete_v)[:, :8]
            h_a = self.proj_a(incomplete_a)[:, :8]
        if complete_for_aux:
            c_l = self.proj_l(text)[:, :8]
            c_v = self.proj_v(vision)[:, :8]
            c_a = self.proj_a(audio)[:, :8]
        else:
            c_l = c_v = c_a = None
        kl_loss, proxy, weights = self.generate_proxy_modality(
            h_l, h_v, h_a, c_l, c_v, c_a)
        if self.use_coverage_gate:
            if valid_lengths is None or valid_lengths.shape != (len(text),):
                raise ValueError("coverage_gate requires B effective lengths")
            positions = torch.arange(50, device=text.device)[None, :]
            content = (positions >= 1) & (positions < valid_lengths[:, None] - 1)
            coverage = (observed_mask & content[:, None, :]).sum(-1).float()
            coverage = coverage / content.sum(-1).clamp_min(1).float()[:, None]
            # Only calibrate the original uncertainty weights in the author's
            # crossmodal injection. The proxy representation remains original.
            correction = self.coverage_beta * coverage.clamp_min(1e-4).log().T[:, :, None, None]
            weights = torch.softmax(weights.float().clamp_min(1e-8).log() + correction, dim=0)
            weights = weights.to(proxy.dtype)
        fused = self.crossmodal_encoder(self.GRL(proxy), h_l, h_a, h_v, weights)
        summary = fused.mean(dim=1)
        regression = self.fc2(self.dropout(F.relu(self.fc1(summary))))
        cls_logits = self.cls_head(summary)
        reconstruction = target = None
        if complete_for_aux:
            r_a = self.reconstructor[0](h_a)[:, :8]
            r_v = self.reconstructor[1](h_v)[:, :8]
            r_l = self.reconstructor[2](h_l)[:, :8]
            reconstruction = torch.cat([r_a, r_v, r_l], dim=1)
            target = torch.cat([c_a, c_v, c_l], dim=1)
        return {"cls_logits": cls_logits, "logits_c": regression,
                "kl_loss": kl_loss, "rec_feats": reconstruction,
                "complete_feats": target, "uncertainty_weights_LVA": weights,
                "coverage_beta": self.coverage_beta if self.use_coverage_gate else None}
