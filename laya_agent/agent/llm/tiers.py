"""Model-tier resolution for the dual-tier hosted LLM client.

Big Tier (default: ``nemotron-3.5-super-vl-120b-a12b``) and Small Tier (a
7B/8B instruction model) are parameterized via ``configs/arms.yaml`` and
overridable through ``BIG_TIER_MODEL`` / ``SMALL_TIER_MODEL`` environment
variables. Environment variables take precedence over YAML values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class Tier(str, Enum):
    BIG = "big"
    SMALL = "small"


@dataclass(frozen=True)
class TierModel:
    tier: Tier
    model: str
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0


def resolve_tier_model(
    config: Dict[str, Any],
    tier: Tier,
    env_var: str,
    config_key: str,
) -> TierModel:
    """Resolve a single tier's model and pricing from config + env overrides.

    Missing pricing defaults to 0.0 (populated later by Milestone 4 cost
    estimation).
    """
    pricing = (config.get("pricing") or {}).get(tier.value, {})
    model = os.environ.get(env_var) or config.get(config_key)
    if not model:
        raise ValueError(
            f"No model resolved for tier '{tier.value}': set {env_var} or "
            f"config key '{config_key}'."
        )
    return TierModel(
        tier=tier,
        model=str(model),
        input_price_per_1m=float(pricing.get("input_per_1m", 0.0)),
        output_price_per_1m=float(pricing.get("output_per_1m", 0.0)),
    )


def resolve_tiers(config: Dict[str, Any]) -> Dict[Tier, TierModel]:
    """Resolve both Big and Small tier models from the arms config."""
    big = resolve_tier_model(
        config, Tier.BIG, "BIG_TIER_MODEL", "BIG_TIER_MODEL"
    )
    small = resolve_tier_model(
        config, Tier.SMALL, "SMALL_TIER_MODEL", "SMALL_TIER_MODEL"
    )
    return {Tier.BIG: big, Tier.SMALL: small}


__all__ = ["Tier", "TierModel", "resolve_tier_model", "resolve_tiers"]
