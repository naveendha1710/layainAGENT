"""Configuration loader for the LAYA experiment arms matrix.

Loads ``configs/arms.yaml`` and resolves environment-variable placeholders of the
form ``${VAR_NAME}`` so that model IDs, API keys, and pricing can be injected
without touching YAML. Environment variables take precedence over YAML values.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_placeholders(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with ``os.environ`` values."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))  # leave unresolved as-is
        return _ENV_PLACEHOLDER.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v) for v in value]
    return value


def load_arms_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the arms configuration YAML and resolve env var placeholders.

    Parameters
    ----------
    path:
        Explicit path to ``arms.yaml``. Defaults to the ``LAYA_CONFIG_PATH``
        environment variable, then ``configs/arms.yaml`` relative to this
        package's parent directory (the repo root).
    """
    if path is None:
        path = os.environ.get("LAYA_CONFIG_PATH")
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "arms.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _resolve_placeholders(deepcopy(raw))


def get_arm(config: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    """Return the resolved arm definition dict for ``arm_id``."""
    arms = config.get("arms", {})
    if arm_id not in arms:
        raise KeyError(f"Unknown arm '{arm_id}'. Available: {sorted(arms)}")
    return arms[arm_id]


__all__ = ["load_arms_config", "get_arm"]
