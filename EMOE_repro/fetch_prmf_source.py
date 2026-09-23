"""Fetch the pinned public P-RMF source files, excluding images/checkpoints."""
from pathlib import Path

import requests


REVISION = "769c31b50f173b3f7670404a0939a95388393610"
HERE = Path(__file__).resolve().parent
DESTINATION = HERE / "P_RMF_upstream"
TREE = f"https://api.github.com/repos/hawksilent/P-RMF/git/trees/{REVISION}?recursive=1"
ALLOWED = {".py", ".yaml", ".md", ".txt", ".sh"}


def main():
    session = requests.Session()
    response = session.get(TREE, timeout=30)
    response.raise_for_status()
    tree = response.json()
    assert tree["sha"] == REVISION and not tree.get("truncated")
    count = 0
    for entry in tree["tree"]:
        path = Path(entry["path"])
        if entry["type"] != "blob" or path.suffix not in ALLOWED or "ckpt" in path.parts:
            continue
        url = f"https://raw.githubusercontent.com/hawksilent/P-RMF/{REVISION}/{entry['path']}"
        content = session.get(url, timeout=30)
        content.raise_for_status()
        if len(content.content) != entry["size"]:
            raise ValueError(f"file size mismatch: {entry['path']}")
        target = DESTINATION / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.content)
        count += 1
    (DESTINATION / "UPSTREAM_REVISION.txt").write_text(
        f"https://github.com/hawksilent/P-RMF\n{REVISION}\n", encoding="utf-8")
    print(f"saved {count} files to {DESTINATION}")


if __name__ == "__main__":
    main()
