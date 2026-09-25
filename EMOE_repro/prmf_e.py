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

    def __init__(self, cfg=None, *, missing_embedding=False, coverage_gate=False,
                 zero_padding=False, split_text_head=False,
                 masked_attention=False, coupled_head=False,
                 local_recovery=False, attention_pool=False,
                 unshared_cross=False, nonlinear_cls=False,
                 detach_cls_residual=False):
        super().__init__()
        cfg = prmf_config() if cfg is None else cfg
        self.cfg = cfg
        self.use_missing_embedding = missing_embedding
        self.use_coverage_gate = coverage_gate
        self.zero_padding = zero_padding
        self.split_text_head = split_text_head
        self.masked_attention = masked_attention
        self.coupled_head = coupled_head
        self.local_recovery = local_recovery
        self.attention_pool = attention_pool
        self.unshared_cross = unshared_cross
        self.nonlinear_cls = nonlinear_cls
        self.detach_cls_residual = detach_cls_residual
        if masked_attention and local_recovery:
            raise ValueError("masked_attention and local_recovery are separate ablations")
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
        if coupled_head:
            # A small nonlinear bridge from the regression prediction to the
            # three polarity logits. Zero init reproduces the original head
            # before training; CE can still update the shared valence estimate.
            self.cls_from_reg = nn.Linear(2, 3, bias=False)
            nn.init.zeros_(self.cls_from_reg.weight)
        if split_text_head:
            self.cls_head_text_missing = nn.Linear(regression["input_dim"], 3)
            self.cls_head_text_missing.load_state_dict(self.cls_head.state_dict())
        # E-question variants only. Zero initialization preserves the baseline
        # at step zero and consumes no RNG for the shared author-core layers.
        if missing_embedding:
            self.missing_embedding = nn.Parameter(torch.zeros(3, feature["hidden_dims"][0]))
        if coverage_gate:
            self.coverage_beta = nn.Parameter(torch.zeros(()))
        if local_recovery:
            hidden = feature["hidden_dims"][0]
            self.local_recovery_net = nn.Sequential(
                nn.Conv1d(3 * hidden + 3, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, 3 * hidden, kernel_size=1))
            nn.init.zeros_(self.local_recovery_net[-1].weight)
            nn.init.zeros_(self.local_recovery_net[-1].bias)
        if attention_pool:
            # A zero-init residual to uniform token pooling: the initial
            # forward exactly matches mean pooling while training can learn
            # which of the eight fused proxy tokens to emphasize.
            self.pool_score = nn.Linear(cross["embed_dim"], 1)
            nn.init.zeros_(self.pool_score.weight)
            nn.init.zeros_(self.pool_score.bias)
        if unshared_cross:
            # The upstream encoder appends the *same* layer four times. Start
            # four independent copies at identical weights, so step-zero
            # outputs match the shared implementation exactly.
            original = self.crossmodal_encoder.encoderlayer
            self.crossmodal_encoder.layers = nn.ModuleList(
                deepcopy(original) for _ in range(self.crossmodal_encoder.num_layers))
            del self.crossmodal_encoder.encoderlayer
        if nonlinear_cls:
            self.cls_residual = nn.Sequential(
                nn.Linear(regression["input_dim"], regression["input_dim"]),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(regression["input_dim"], 3))
            nn.init.zeros_(self.cls_residual[-1].weight)
            nn.init.zeros_(self.cls_residual[-1].bias)

    def forward(self, text, audio, vision, observed_mask, complete_for_aux=False,
                valid_lengths=None, complete_text=None, return_features=False):
        if text.ndim != 3 or tuple(text.shape[1:]) != (50, 768):
            raise ValueError("Text must be B×50×768 aligned continuous features")
        if audio.shape != (len(text), 50, 74) or vision.shape != (len(text), 50, 35):
            raise ValueError("Audio/Vision must be B×50×74 and B×50×35")
        if observed_mask.shape != (len(text), 3, 50):
            raise ValueError("observed_mask must be B×3×50, L/V/A order")
        if self.zero_padding:
            if valid_lengths is None or valid_lengths.shape != (len(text),):
                raise ValueError("zero_padding requires B effective lengths")
            positions = torch.arange(50, device=text.device)[None, :]
            valid = positions < valid_lengths[:, None]
            if self.zero_padding == "text_missing":
                content = (positions >= 1) & (positions < valid_lengths[:, None] - 1)
                text_missing = ((~observed_mask[:, 0].bool()) & content).any(dim=1)
                valid = valid | ~text_missing[:, None]
            valid = valid.to(text.dtype)[:, :, None]
            text = text * valid
            audio = audio * valid
            vision = vision * valid
            if complete_text is not None:
                complete_text = complete_text * valid
        # Original P-RMF represents removed features by zero; unlike EMOE it
        # has no explicit missing token. The same input mask is shared by both.
        incomplete_l = text * observed_mask[:, 0, :, None]
        incomplete_v = vision * observed_mask[:, 1, :, None]
        incomplete_a = audio * observed_mask[:, 2, :, None]
        key_masks = None
        complete_key_masks = None
        if self.masked_attention:
            if valid_lengths is None or valid_lengths.shape != (len(text),):
                raise ValueError("masked_attention requires B effective lengths")
            positions = torch.arange(50, device=text.device)[None, :]
            in_length = positions < valid_lengths[:, None]
            key_masks = tuple(observed_mask[:, channel].bool() & in_length
                              for channel in range(3))
            if complete_for_aux:
                complete_key_masks = (
                    in_length,
                    in_length & vision.abs().sum(dim=-1).ne(0),
                    in_length & audio.abs().sum(dim=-1).ne(0),
                )
        local_rec_loss = None
        if self.local_recovery:
            observed = observed_mask.bool()
            projected = (self.proj_l[0](incomplete_l),
                         self.proj_v[0](incomplete_v),
                         self.proj_a[0](incomplete_a))
            local_input = torch.cat(
                [projected[channel] * observed[:, channel, :, None]
                 for channel in range(3)] +
                [observed.transpose(1, 2).to(text.dtype)], dim=-1)
            delta = self.local_recovery_net(
                local_input.transpose(1, 2)).transpose(1, 2).chunk(3, dim=-1)
            restored = tuple(projected[channel] +
                             (~observed[:, channel, :, None]) * delta[channel]
                             for channel in range(3))
            h_l = self.proj_l[1](restored[0])[:, :8]
            h_v = self.proj_v[1](restored[1])[:, :8]
            h_a = self.proj_a[1](restored[2])[:, :8]
            if complete_for_aux:
                if valid_lengths is None:
                    raise ValueError("local recovery loss requires valid lengths")
                positions = torch.arange(50, device=text.device)[None, :]
                in_length = positions < valid_lengths[:, None]
                source_text = text if complete_text is None else complete_text
                available = torch.stack((
                    in_length,
                    in_length & vision.abs().sum(dim=-1).ne(0),
                    in_length & audio.abs().sum(dim=-1).ne(0)), dim=1)
                supervised_missing = (~observed) & available
                target_projected = (
                    self.proj_l[0](source_text),
                    self.proj_v[0](vision),
                    self.proj_a[0](audio))
                squared = torch.stack([
                    (restored[channel].float() -
                     target_projected[channel].detach().float()).square()
                    for channel in range(3)], dim=1)
                local_rec_loss = (
                    (squared * supervised_missing[:, :, :, None]).sum() /
                    (supervised_missing.sum().clamp_min(1) * squared.shape[-1]))
        elif self.use_missing_embedding:
            def encode(proj, values, mask, channel):
                projected = proj[0](values)
                projected = projected + (~mask).to(projected.dtype)[..., None] * self.missing_embedding[channel]
                return proj[1](projected,
                               key_mask=None if key_masks is None else key_masks[channel])[:, :8]

            h_l = encode(self.proj_l, incomplete_l, observed_mask[:, 0], 0)
            h_v = encode(self.proj_v, incomplete_v, observed_mask[:, 1], 1)
            h_a = encode(self.proj_a, incomplete_a, observed_mask[:, 2], 2)
        else:
            if key_masks is None:
                h_l = self.proj_l(incomplete_l)[:, :8]
                h_v = self.proj_v(incomplete_v)[:, :8]
                h_a = self.proj_a(incomplete_a)[:, :8]
            else:
                h_l = self.proj_l[1](self.proj_l[0](incomplete_l),
                                       key_mask=key_masks[0])[:, :8]
                h_v = self.proj_v[1](self.proj_v[0](incomplete_v),
                                       key_mask=key_masks[1])[:, :8]
                h_a = self.proj_a[1](self.proj_a[0](incomplete_a),
                                       key_mask=key_masks[2])[:, :8]
        if complete_for_aux:
            # When missing tokens are re-encoded by BERT, the corrupted text
            # representation differs at *all* positions. Keep the auxiliary
            # reconstruction target on the original complete representation.
            if complete_key_masks is None:
                c_l = self.proj_l(text if complete_text is None else complete_text)[:, :8]
                c_v = self.proj_v(vision)[:, :8]
                c_a = self.proj_a(audio)[:, :8]
            else:
                source_text = text if complete_text is None else complete_text
                c_l = self.proj_l[1](self.proj_l[0](source_text),
                                       key_mask=complete_key_masks[0])[:, :8]
                c_v = self.proj_v[1](self.proj_v[0](vision),
                                       key_mask=complete_key_masks[1])[:, :8]
                c_a = self.proj_a[1](self.proj_a[0](audio),
                                       key_mask=complete_key_masks[2])[:, :8]
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
        if self.attention_pool:
            pool_weights = torch.softmax(self.pool_score(fused).float(), dim=1)
            summary = (fused * pool_weights.to(fused.dtype)).sum(dim=1)
        else:
            summary = fused.mean(dim=1)
        regression = self.fc2(self.dropout(F.relu(self.fc1(summary))))
        cls_logits = self.cls_head(summary)
        if self.nonlinear_cls:
            cls_logits = cls_logits + self.cls_residual(
                summary.detach() if self.detach_cls_residual else summary)
        if self.coupled_head:
            valence = torch.cat((regression, regression.abs()), dim=1)
            cls_logits = cls_logits + self.cls_from_reg(valence)
        if self.split_text_head:
            if valid_lengths is None or valid_lengths.shape != (len(text),):
                raise ValueError("split_text_head requires B effective lengths")
            positions = torch.arange(50, device=text.device)[None, :]
            content = (positions >= 1) & (positions < valid_lengths[:, None] - 1)
            text_missing = ((~observed_mask[:, 0].bool()) & content).any(dim=1)
            missing_logits = self.cls_head_text_missing(summary)
            cls_logits = torch.where(text_missing[:, None], missing_logits, cls_logits)
        reconstruction = target = None
        if complete_for_aux:
            r_a = self.reconstructor[0](h_a)[:, :8]
            r_v = self.reconstructor[1](h_v)[:, :8]
            r_l = self.reconstructor[2](h_l)[:, :8]
            reconstruction = torch.cat([r_a, r_v, r_l], dim=1)
            target = torch.cat([c_a, c_v, c_l], dim=1)
        result = {"cls_logits": cls_logits, "logits_c": regression,
                  "kl_loss": kl_loss, "rec_feats": reconstruction,
                  "complete_feats": target, "uncertainty_weights_LVA": weights,
                  "coverage_beta": self.coverage_beta if self.use_coverage_gate else None,
                  "local_rec_loss": local_rec_loss}
        if return_features:
            result["fused_summary"] = summary
            result["unimodal_summary_LVA"] = torch.cat(
                (h_l.mean(dim=1), h_v.mean(dim=1), h_a.mean(dim=1)), dim=1)
        return result
