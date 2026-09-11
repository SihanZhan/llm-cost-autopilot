"""Load routing.yaml and resolve a complexity tier to a registry model.

Kept deliberately dumb: read the YAML, validate every tier is covered and
every model_id exists in the registry, done. The actual "call the classifier,
look up the tier, send the request" wiring is router/API work (Phase 3+).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from classifier.tiers import TIERS
from models import ModelConfig, get_model

ROUTING_FILE = Path(__file__).parent.parent / "routing.yaml"


def load_routing_config(path: Path = ROUTING_FILE) -> dict[int, str]:
    """Return the tier -> model_id mapping from ``path``, validated."""
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    mapping = raw.get("routing", {})
    config = {int(tier): model_id for tier, model_id in mapping.items()}

    missing = set(TIERS) - set(config)
    if missing:
        raise ValueError(f"{path.name} is missing tier(s): {sorted(missing)}")
    for tier, model_id in config.items():
        get_model(model_id)  # raises KeyError if model_id isn't registered

    return config


def model_for_tier(tier: int, path: Path = ROUTING_FILE) -> ModelConfig:
    """Return the registry ``ModelConfig`` routed to for ``tier``."""
    config = load_routing_config(path)
    if tier not in config:
        raise ValueError(f"no routing entry for tier {tier}; expected one of {TIERS}")
    return get_model(config[tier])


if __name__ == "__main__":  # quick manual check: python -m classifier.routing
    for tier in TIERS:
        model = model_for_tier(tier)
        print(f"tier {tier} -> {model.model_id} ({model.provider}, {model.quality_tier})")
