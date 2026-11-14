import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegis_mmu.config import AegisSettings  # noqa: E402
from aegis_mmu.db.store import AegisStore  # noqa: E402
from aegis_mmu.mmu.pipeline import SemanticMMU  # noqa: E402

SAMPLES = ROOT / "tests" / "samples"
DATASETS = ROOT / "evals" / "datasets"


@pytest.fixture()
def settings() -> AegisSettings:
    return AegisSettings(db_path=":memory:")


@pytest.fixture()
def store(settings) -> AegisStore:
    s = AegisStore(":memory:", settings=settings)
    yield s
    s.close()


@pytest.fixture()
def mmu(store, settings) -> SemanticMMU:
    return SemanticMMU(store, settings=settings)


@pytest.fixture(scope="session")
def codebase() -> list[tuple[str, str]]:
    root = DATASETS / "codebase"
    return [(str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()]
