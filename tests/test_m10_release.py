from __future__ import annotations

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_m10_version_is_synchronized() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    init_text = (ROOT / "src/ui_rag/__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, flags=re.MULTILINE)
    assert match is not None
    assert pyproject["project"]["version"] == "0.10.0.1"
    assert match.group(1) == "0.10.0.1"


def test_m10_portfolio_docs_exist() -> None:
    names = [
        "PORTFOLIO_CASE_STUDY.md",
        "ARCHITECTURE.md",
        "BENCHMARKS.md",
        "DEMO.md",
        "INTERVIEW_GUIDE.md",
        "REPRODUCIBILITY.md",
        "RELEASE_CHECKLIST.md",
        "M10_PORTFOLIO_RELEASE.md",
    ]
    assert all((ROOT / "docs" / name).is_file() for name in names)


def test_readme_preserves_evaluation_caveats() -> None:
    text = (ROOT / "README.md").read_text()
    assert "16.73%" in text
    assert "77.05%" in text
    assert "target-conditioned" in text
    assert "automated semantic scores were discarded" in text
    assert "12-case" in text


def test_public_portfolio_docs_do_not_contain_local_user_path() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "docs/PORTFOLIO_CASE_STUDY.md",
        ROOT / "docs/ARCHITECTURE.md",
        ROOT / "docs/BENCHMARKS.md",
        ROOT / "docs/DEMO.md",
        ROOT / "docs/INTERVIEW_GUIDE.md",
        ROOT / "docs/REPRODUCIBILITY.md",
        ROOT / "docs/RELEASE_CHECKLIST.md",
    ]
    text = "\n".join(path.read_text() for path in paths)
    assert "/Users/" not in text
    assert "mishka" not in text.lower()
    assert "bashkirka" not in text.lower()


def test_readme_links_to_core_portfolio_docs() -> None:
    text = (ROOT / "README.md").read_text()
    for doc in (
        "docs/PORTFOLIO_CASE_STUDY.md",
        "docs/ARCHITECTURE.md",
        "docs/BENCHMARKS.md",
        "docs/DEMO.md",
        "docs/INTERVIEW_GUIDE.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RELEASE_CHECKLIST.md",
    ):
        assert doc in text
