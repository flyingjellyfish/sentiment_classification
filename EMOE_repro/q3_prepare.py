"""Audit E-question attachment 4 and convert NumPy-2 pickles for the Q3 runtime.

Run with the bundled NumPy-2 Python. The training venv has NumPy 1.x and
cannot unpickle attachment 4's ``numpy._core`` objects directly.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT2 = ROOT / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl"
ATTACHMENT4 = (ROOT / "E题数据" / "附件4-可解释专项视频样本与特征文件"
               / "附件4-可解释专项视频样本与特征文件" / "对齐版本")
OUT = Path(__file__).resolve().parent / "results" / "q3"


def boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos:pos + 4], "big")
        typ = data[pos + 4:pos + 8]
        header = 8
        if size == 1:
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"Invalid MP4 box {typ!r} at {pos}")
        yield typ, pos + header, pos + size
        pos += size


def children(data: bytes, region: tuple[int, int], kind: bytes):
    return [(a, b) for typ, a, b in boxes(data, *region) if typ == kind]


def media_metadata(path: Path):
    data = path.read_bytes()
    top = (0, len(data))
    moov = children(data, top, b"moov")
    if len(moov) != 1:
        raise ValueError(f"Expected one moov atom: {path}")
    mvhd = children(data, moov[0], b"mvhd")[0]
    payload = data[slice(*mvhd)]
    if payload[0] == 0:
        scale, ticks = struct.unpack_from(">II", payload, 12)
    elif payload[0] == 1:
        scale, ticks = struct.unpack_from(">IQ", payload, 20)
    else:
        raise ValueError("Unknown mvhd version")
    duration = ticks / scale
    frames = None
    for trak in children(data, moov[0], b"trak"):
        mdia = children(data, trak, b"mdia")[0]
        hdlr = data[slice(*children(data, mdia, b"hdlr")[0])]
        if hdlr[8:12] != b"vide":
            continue
        stbl = children(data, children(data, mdia, b"minf")[0], b"stbl")[0]
        stsz = data[slice(*children(data, stbl, b"stsz")[0])]
        frames = struct.unpack_from(">I", stsz, 8)[0]
        break
    if not (duration > 0 and frames and frames > 0):
        raise ValueError(f"Cannot read positive duration/frame count: {path}")
    return {"duration_s": round(duration, 6), "video_frames": frames,
            "mean_fps": round(frames / duration, 6)}


def ids(values):
    return {str(v[0] if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 1 else v)
            for v in values}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ATTACHMENT2.open("rb") as stream:
        source = pickle.load(stream)
    expected = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35),
                "text_bert": (3, 50)}
    splits = {}
    source_ids = set()
    for split in ("train", "valid", "test"):
        item = source[split]
        n = len(item["id"])
        shapes = {k: list(np.asarray(item[k]).shape) for k in expected}
        assert all(tuple(shapes[k]) == (n, *shape) for k, shape in expected.items())
        split_ids = ids(item["id"])
        assert len(split_ids) == n, f"duplicate IDs in {split}"
        splits[split] = {"count": n, "shapes": shapes, "id_count": len(split_ids),
                         "first_id": sorted(split_ids)[0]}
        source_ids |= split_ids

    paths = sorted(ATTACHMENT4.glob("*.pkl"))
    assert len(paths) == 20
    arrays = {k: [] for k in expected}
    sample_ids, raw_texts, durations, frames, rows = [], [], [], [], []
    for path in paths:
        with path.open("rb") as stream:
            item = pickle.load(stream)
        assert set(item) == set(expected) | {"raw_text", "id"}
        sid = str(item["id"])
        assert sid == path.stem, (sid, path)
        video = ATTACHMENT4 / "videos" / f"{sid}.mp4"
        assert video.is_file() and video.stat().st_size > 0
        media = media_metadata(video)
        for key, shape in expected.items():
            value = np.asarray(item[key])
            assert value.shape == shape, (path, key, value.shape)
            if key != "text_bert":
                assert np.isfinite(value).all(), (path, key)
            arrays[key].append(value.astype(np.int64 if key == "text_bert" else np.float32))
        length = int(np.asarray(item["text_bert"])[1].sum())
        assert 3 <= length <= 50
        sample_ids.append(sid)
        raw_texts.append(str(item["raw_text"]))
        durations.append(media["duration_s"])
        frames.append(media["video_frames"])
        rows.append({"id": sid, "pkl": str(path), "video": str(video),
                     "pkl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "valid_token_length": length,
                     "zero_rows_audio": int((np.asarray(item["audio"]) == 0).all(axis=1).sum()),
                     "zero_rows_vision": int((np.asarray(item["vision"]) == 0).all(axis=1).sum()),
                     **media})
    assert len(set(sample_ids)) == len(sample_ids)
    assert len(list((ATTACHMENT4 / "videos").glob("*.mp4"))) == len(paths)
    output = OUT / "attachment4_aligned50.npz"
    np.savez_compressed(output, **{k: np.stack(v) for k, v in arrays.items()},
                        id=np.asarray(sample_ids), raw_text=np.asarray(raw_texts),
                        duration_s=np.asarray(durations, dtype=np.float64),
                        video_frames=np.asarray(frames, dtype=np.int64))
    report = {"feature_version": "aligned_50", "attachment2": splits,
              "attachment4_count": len(rows), "attachment4_overlap_with_attachment2_ids":
              sorted(set(sample_ids) & source_ids), "samples": rows,
              "conversion_output": str(output)}
    (OUT / "input_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": {k: v["count"] for k, v in splits.items()},
                      "attachment4_count": len(rows), "duration_range_s":
                      [min(durations), max(durations)], "id_overlap":
                      report["attachment4_overlap_with_attachment2_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
