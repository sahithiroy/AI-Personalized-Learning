"""Test setup: isolated store dirs, offline embedder and mock LLMs (no API calls)."""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="plrs_test_"))

_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
_cfg["paths"] = {"store_dir": str(_TMP / "store"), "models_dir": str(_TMP / "models")}
_cfg["ekt"]["epochs"] = 2
(_TMP / "config.yaml").write_text(yaml.safe_dump(_cfg), encoding="utf-8")

os.environ["PLRS_CONFIG"] = str(_TMP / "config.yaml")
os.environ["PLRS_EMBEDDER"] = "hashing"
for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
    os.environ[key] = ""  # empty -> mock provider; load_dotenv() will not override it

CURRICULUM = ROOT / "data" / "sample" / "curriculum_c_programming.txt"


@pytest.fixture(scope="session")
def kb():
    from plrs.rag import CurriculumKB

    kb = CurriculumKB.ingest([CURRICULUM])
    kb.extract_concepts()
    kb.save()
    return kb


@pytest.fixture(scope="session")
def sample_data():
    from plrs.data import generate_sample_data

    return generate_sample_data(_TMP / "data", n_historical=80, n_current=20, seed=3)


@pytest.fixture(scope="session")
def tracer(sample_data):
    from plrs.data import load_records
    from plrs.kt import train_kt

    tracer, _ = train_kt(load_records(sample_data["historical"]), "ekt", epochs=2)
    return tracer
