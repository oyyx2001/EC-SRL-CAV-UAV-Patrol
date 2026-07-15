from __future__ import annotations

import argparse
from pathlib import Path
import time
import unittest

import pandas as pd


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
        self._started = {}

    def startTest(self, test):
        self._started[str(test)] = time.perf_counter()
        super().startTest(test)

    def _row(self, test, status, detail=""):
        name = str(test)
        self.rows.append({
            "test": name,
            "module": test.__class__.__module__,
            "status": status,
            "duration_s": time.perf_counter() - self._started.get(name, time.perf_counter()),
            "detail": detail,
        })

    def addSuccess(self, test):
        self._row(test, "passed"); super().addSuccess(test)

    def addFailure(self, test, err):
        self._row(test, "failed", self._exc_info_to_string(err, test)); super().addFailure(test, err)

    def addError(self, test, err):
        self._row(test, "error", self._exc_info_to_string(err, test)); super().addError(test, err)

    def addSkip(self, test, reason):
        self._row(test, "skipped", reason); super().addSkip(test, reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/stage4_joint_plan_generation")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(result.rows)
    frame.to_csv(out / "unit_test_results.csv", index=False)
    frame.groupby(["module", "status"]).size().rename("count").reset_index().to_csv(
        out / "unit_test_summary.csv", index=False
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
