"""Configuration loader. All paths are resolved relative to the project root."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def load_config(path=None) -> dict:
    path = Path(path or os.getenv("MEDIRAG_CONFIG") or ROOT / "configs" / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = val if os.path.isabs(val) else str((ROOT / val).resolve())
    if os.getenv("MEDIRAG_LLM_BACKEND"):
        cfg["llm"]["backend"] = os.environ["MEDIRAG_LLM_BACKEND"]
    if os.getenv("MEDIRAG_LLM_MODEL"):
        cfg["llm"]["model_name"] = os.environ["MEDIRAG_LLM_MODEL"]
    if os.getenv("MEDIRAG_TEST_MODE") == "1":  # no downloads; for smoke tests / demos of plumbing
        cfg["embedding"]["backend"] = "hashing"
        cfg["llm"]["backend"] = "extractive"
        cfg["reranker"]["enabled"] = False
        cfg["retrieval"]["min_evidence_cosine"] = 0.05
    return cfg
