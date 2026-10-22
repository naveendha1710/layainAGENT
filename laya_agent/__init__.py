"""LAYA Agent Research Project package.

A sub-40 ms confidence-gated modular control layer wrapped around a ReAct-style
tool-using LLM agent backed by a local vector policy store.

All five decision gates (scorer, filter, router, firewall, supervisor) are hooks
integrated into the main ReAct loop and are active or inactive strictly via
``configs/arms.yaml`` boolean/mode flags ("build once, toggle via config").
"""

__version__ = "0.5.0"
