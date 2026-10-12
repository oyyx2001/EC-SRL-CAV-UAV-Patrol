"""License and input-integrity tests; unit fixtures are not geographic data."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.check_data_assets import CASES, ROOT, inspect_assets


@pytest.fixture
def unit_manifest(tmp_path):
    # Deliberately tiny arbitrary bytes, not real maps or synthetic paper data.
    manifest = {"asset_sha256": {}}
    for case in CASES:
        relative = f"data/{case}/unit_fixture.bin"
        payload = case.encode()
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        manifest["asset_sha256"][case] = {relative: hashlib.sha256(payload).hexdigest()}
    directory = tmp_path / "reproduction"
    directory.mkdir()
    (directory / "evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest


def test_standard_mit_terms_and_copyright_notice():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 EC-SRL-CAV-UAV-Patrol contributors" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED" in license_text
    assert "The above copyright notice and this permission notice shall be included" in license_text
    assert license_text.rstrip().endswith("SOFTWARE.")
    scope = (ROOT / "LICENSE_SCOPE.md").read_text(encoding="utf-8")
    assert "Not relicensed under MIT" in scope
    for case in CASES:
        assert f"cs_rdqn_{case}.pt" in scope


def test_current_index_contains_no_withheld_geographic_files():
    if not (ROOT / ".git").exists():
        pytest.skip("Source archive has no Git index to inspect")
    result = subprocess.run(["git", "ls-files", "--", "data", "truck_uav_modules/data"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    allowed = {"data/README.md", "truck_uav_modules/data/README.md"}
    assert set(result.stdout.splitlines()).issubset(allowed)
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/data/*" in ignored and "/truck_uav_modules/data/*" in ignored


def test_weights_unchanged_after_license_notice():
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    for row in manifest:
        path = ROOT / row["artifact"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_input_preflight_reports_matches(unit_manifest):
    root, _ = unit_manifest
    rows = inspect_assets(root)
    assert len(rows) == 2 and all(row["status"] == "MATCH" for row in rows)


def test_input_preflight_reports_missing(unit_manifest):
    root, manifest = unit_manifest
    relative = next(iter(manifest["asset_sha256"][CASES[0]]))
    (root / relative).unlink()
    rows = inspect_assets(root, [CASES[0]])
    assert rows[0]["status"] == "MISSING" and rows[0]["actual_sha256"] is None


def test_input_preflight_reports_mismatch(unit_manifest):
    root, manifest = unit_manifest
    relative = next(iter(manifest["asset_sha256"][CASES[0]]))
    (root / relative).write_bytes(b"different unit fixture")
    assert inspect_assets(root, [CASES[0]])[0]["status"] == "MISMATCH"


def test_input_preflight_rejects_path_escape(unit_manifest):
    root, manifest = unit_manifest
    manifest["asset_sha256"][CASES[0]] = {"../outside.bin": "unused"}
    (root / "reproduction/evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        inspect_assets(root, [CASES[0]])


def test_input_preflight_rejects_unknown_case(unit_manifest):
    root, _ = unit_manifest
    with pytest.raises(ValueError, match="Unknown case"):
        inspect_assets(root, ["not_a_case"])
