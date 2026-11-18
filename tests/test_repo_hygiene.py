"""Repository hygiene: no data or generated artefacts tracked, and no source file hidden
by .gitignore (a bare 'data/' rule once excluded src/corelog/data/)."""
import shutil
import subprocess

import pytest

from corelog.config import REPO_ROOT

FORBIDDEN_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".cr2", ".cr3", ".nef", ".dng", ".npy", ".npz",
                 ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".xlsx", ".xls", ".parquet", ".h5"}


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def candidate_files():
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(_git("ls-files").split())
    untracked = set(_git("ls-files", "--others", "--exclude-standard").split())
    return sorted(tracked | untracked)


def test_no_data_or_checkpoints_in_git(candidate_files):
    bad = [f for f in candidate_files if any(f.lower().endswith(e) for e in FORBIDDEN_EXT)]
    assert not bad, f"data/image/checkpoint files would be committed: {bad}"


def test_no_large_files_in_git(candidate_files):
    big = [f for f in candidate_files if (REPO_ROOT / f).is_file() and (REPO_ROOT / f).stat().st_size > 1_000_000]
    assert not big, f"files > 1 MB would be committed: {big}"


def test_no_source_or_test_file_is_ignored():
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = [str(p.relative_to(REPO_ROOT)) for d in ("src", "tests", "configs", "docs")
             for p in (REPO_ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".yaml", ".md", ".csv"}]
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", *files], capture_output=True, text=True)
    ignored = [f for f in r.stdout.split() if not f.startswith("configs/lab") and not f.startswith("configs/local")]
    assert not ignored, f".gitignore hides project files: {ignored}"
