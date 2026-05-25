#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL shard manifests into one manifest.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as writer:
        for pattern in args.inputs:
            paths = sorted(Path().glob(pattern))
            if not paths:
                paths = [Path(pattern)]
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open("r", encoding="utf-8") as reader:
                    for line in reader:
                        if line.strip():
                            writer.write(line if line.endswith("\n") else line + "\n")
                            total += 1
    print(f"Wrote {total} rows to {output}")


if __name__ == "__main__":
    main()

