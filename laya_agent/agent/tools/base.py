"""Deterministic tool interface and registry for the execution layer.

Tools are plain callables registered by name. The ReAct loop invokes them by
name with parsed JSON arguments, and results flow back into the loop as
observations (optionally through the Context Firewall gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]


class ToolRegistry:
    """Registry of named, deterministic tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, fn=fn)

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(
                f"Unknown tool '{name}'. Registered: {sorted(self._tools)}"
            )
        return self._tools[name].fn(**kwargs)

    def specs(self) -> Dict[str, ToolSpec]:
        return dict(self._tools)


__all__ = ["ToolRegistry", "ToolSpec"]
