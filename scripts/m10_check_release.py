from __future__ import annotations

import json
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = (
    "docs/PORTFOLIO_CASE_STUDY.md",
    "docs/ARCHITECTURE.md",
    "docs/BENCHMARKS.md",
    "docs/DEMO.md",
    "docs/INTERVIEW_GUIDE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/M10_PORTFOLIO_RELEASE.md",
    "docs/M9_FINAL_RESULTS.md",
    "docs/m10_reference_summary.csv",
    "docs/m10_reference_run.json",
)

PORTFOLIO_DOCS = (
    "README.md",
    "docs/PORTFOLIO_CASE_STUDY.md",
    "docs/ARCHITECTURE.md",
    "docs/BENCHMARKS.md",
    "docs/DEMO.md",
    "docs/INTERVIEW_GUIDE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/M10_PORTFOLIO_RELEASE.md",
)

REQUIRED_README_MARKERS = (
    "## Key results",
    "## Final architecture",
    "## Evaluation integrity",
    "## Known limitations",
    "## Portfolio documentation",
)

FORBIDDEN_PUBLIC_PATTERNS = (
    re.compile(r"/Users/[^/\s`]+"),
    re.compile(r"MacBook-Air-[A-Za-z0-9_-]+"),
    re.compile(r"mishka", re.IGNORECASE),
    re.compile(r"bashkirka", re.IGNORECASE),
)


def _project_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def _package_version() -> str:
    text = (ROOT / "src/ui_rag/__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not parse src/ui_rag/__init__.py version")
    return match.group(1)


def main() -> None:
    missing = [path for path in REQUIRED_DOCS if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing M10 portfolio docs: {missing}")

    project_version = _project_version()
    package_version = _package_version()
    if project_version != package_version:
        raise SystemExit(
            f"Version mismatch: pyproject={project_version}, package={package_version}"
        )
    if project_version != "0.10.0.1":
        raise SystemExit(f"Unexpected M10 version: {project_version}")

    readme = (ROOT / "README.md").read_text()
    missing_markers = [marker for marker in REQUIRED_README_MARKERS if marker not in readme]
    if missing_markers:
        raise SystemExit(f"README is missing required portfolio sections: {missing_markers}")

    public_leaks: list[str] = []
    for relative in PORTFOLIO_DOCS:
        text = (ROOT / relative).read_text()
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            if pattern.search(text):
                public_leaks.append(f"{relative}: {pattern.pattern}")
    if public_leaks:
        raise SystemExit(
            f"Public portfolio docs contain local identity/path leakage: {public_leaks}"
        )

    gitignore = (ROOT / ".gitignore").read_text()
    required_ignores = ("__pycache__/", "*.py[cod]", ".pytest_cache/", ".ruff_cache/", ".DS_Store")
    missing_ignores = [item for item in required_ignores if item not in gitignore]
    if missing_ignores:
        raise SystemExit(f"Missing generated-file ignores: {missing_ignores}")

    # Guard the most important portfolio caveats so later README edits do not
    # accidentally convert diagnostics into production claims.
    claim_contract = {
        "final_recall_at_10": "16.73%",
        "target_conditioned_diagnostic": "77.05%",
        "judge_calibration_failure": "0%",
        "warm_search_latency": "171 ms",
        "human_audit_size": "12-case",
    }
    for label, value in claim_contract.items():
        if value not in readme:
            raise SystemExit(f"README claim contract missing {label}={value}")

    # Validate relative Markdown links in the recruiter-facing README.
    for target in re.findall(r"\[[^\]]+\]\(([^)]+\.md)\)", readme):
        if target.startswith(("http://", "https://")):
            continue
        if not (ROOT / target).is_file():
            raise SystemExit(f"README contains a broken Markdown link: {target}")

    for relative in PORTFOLIO_DOCS:
        text = (ROOT / relative).read_text()
        if "TODO" in text or "TBD" in text:
            raise SystemExit(f"Portfolio doc still contains TODO/TBD marker: {relative}")

    result = {
        "milestone": "M10",
        "version": project_version,
        "required_docs": len(REQUIRED_DOCS),
        "readme_markers": len(REQUIRED_README_MARKERS),
        "public_identity_path_scrub": True,
        "generated_file_ignores": True,
        "portfolio_claim_contract": True,
        "ml_serving_semantics_changed": False,
    }
    print(json.dumps(result, indent=2))
    print("M10 portfolio release check PASSED")


if __name__ == "__main__":
    main()
