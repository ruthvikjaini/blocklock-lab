"""blocklock-lab: reproducing BlockLock's batch-geometry nondeterminism and
its fixed-geometry fix, on CPU with real bf16 arithmetic."""

from . import audit, bf16, model, policies, reductions

__all__ = ["audit", "bf16", "model", "policies", "reductions"]
