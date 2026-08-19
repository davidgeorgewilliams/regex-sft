"""Pre-download Qwen3-4B weights so the rest of the pipeline can run offline.

Both variants land in models/original/<repo-name>/. Downloads are resumable:
re-running skips files already present and complete.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
]

DEST_ROOT = Path(__file__).resolve().parent.parent / "models" / "original"

# Skip alternate-format weights; we only need the safetensors + tokenizer/config.
IGNORE = ["*.gguf", "*.pth", "*.bin", "original/*"]


def main() -> int:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    failed = []

    for repo in REPOS:
        dest = DEST_ROOT / repo.split("/")[-1]
        print(f"\n=== {repo} -> {dest}", flush=True)
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                ignore_patterns=IGNORE,
                max_workers=8,
            )
        except Exception as exc:
            # Keep going so one bad repo does not block the other download.
            print(f"FAILED {repo}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failed.append(repo)
        else:
            print(f"OK {repo}", flush=True)

    if failed:
        print(f"\nDONE WITH FAILURES: {failed}", file=sys.stderr, flush=True)
        return 1
    print("\nALL DOWNLOADS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
