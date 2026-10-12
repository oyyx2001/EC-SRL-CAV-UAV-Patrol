"""Check core scenario-input identity, without downloading or changing simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("case1_xinjiekou", "case2_baijiahu")


def inspect_assets(root: Path, cases=CASES) -> list[dict]:
    root = Path(root).resolve()
    manifest = json.loads((root / "reproduction/evaluation_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        if case not in CASES:
            raise ValueError(f"Unknown case: {case}")
        for relative, expected in manifest["asset_sha256"][case].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Asset path must remain inside the repository root")
            status, actual = "MISSING", None
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                actual = digest.hexdigest()
                status = "MATCH" if actual == expected else "MISMATCH"
            rows.append(dict(case=case, path=relative, status=status,
                             expected_sha256=expected, actual_sha256=actual))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES + ("all",), default="all")
    args = parser.parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    rows = inspect_assets(ROOT, cases)
    for row in rows:
        print(f"{row['status']:8s} {row['case']} {row['path']}")
    print("Checksum identity is not a data-use or redistribution permission.")
    if any(row["status"] != "MATCH" for row in rows):
        print("Original-case inputs are intentionally withheld from the public release. "
              "Obtain authorized matching assets; see DATA_AVAILABILITY.md. "
              "No download, replacement data, training or evaluation was performed.")
        return 1
    print("All core inputs match the recorded evaluation assets. Verify your access rights before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
