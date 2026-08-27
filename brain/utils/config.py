"""Configuration loader — YAML + .env with Pydantic validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_config: dict | None = None


def _load_agents_yaml(config_path: str) -> dict:
    """Load agents.yaml from the same directory as config.yaml."""
    agents_path = Path(config_path).parent / "agents.yaml"
    try:
        with open(agents_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def load_config(config_path: str = "config/config.yaml", env_path: str = "config/.env") -> dict:
    global _config
    if _config is not None:
        return _config

    load_dotenv(env_path)

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    # Merge agents.yaml — per-agent enable/feature flags
    agents_cfg = _load_agents_yaml(config_path)
    if agents_cfg:
        cfg["agents"] = agents_cfg

    # Allow env vars to override config values
    cfg.setdefault("brain", {})
    cfg["brain"]["name"] = os.getenv("BRAIN_NAME", cfg["brain"].get("name", "Brain"))

    cfg.setdefault("llm", {})
    if os.getenv("ANTHROPIC_API_KEY"):
        cfg["llm"]["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("OPENROUTER_API_KEY"):
        cfg["llm"]["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("GOOGLE_AI_API_KEY"):
        cfg["llm"]["google_ai_api_key"] = os.getenv("GOOGLE_AI_API_KEY")

    cfg.setdefault("audio", {}).setdefault("output", {})
    if os.getenv("ELEVENLABS_API_KEY"):
        cfg["audio"]["output"]["elevenlabs_api_key"] = os.getenv("ELEVENLABS_API_KEY")

    cfg.setdefault("vision", {}).setdefault("moondream", {})
    if os.getenv("MOONDREAM_API_KEY"):
        cfg["vision"]["moondream"]["api_key"] = os.getenv("MOONDREAM_API_KEY")

    cfg["brain"]["mock_hardware"] = (
        os.getenv("MOCK_HARDWARE", "false").lower() == "true"
    )
    cfg["brain"]["mock_llm"] = (
        os.getenv("MOCK_LLM", "false").lower() == "true"
    )

    _config = cfg
    return _config


def agent_enabled(agent_name: str) -> bool:
    """Return True if the agent is enabled in agents.yaml (default: True)."""
    cfg = load_config()
    return cfg.get("agents", {}).get(agent_name, {}).get("enabled", True)


def agent_feature(agent_name: str, feature: str, default: bool = True) -> bool:
    """Return True if a specific feature is enabled for an agent (default: True)."""
    cfg = load_config()
    return cfg.get("agents", {}).get(agent_name, {}).get("features", {}).get(feature, default)


def get(key_path: str, default: Any = None) -> Any:
    cfg = load_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val
