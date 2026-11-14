"""Serving-process runtime policy."""

from __future__ import annotations

import gc


def tune_gc() -> None:
    """Latency policy for long-lived serving processes (proxy / MCP server).

    Everything allocated during start-up and index warming (engine projection, vector mirrors,
    parsed mipmaps) is long-lived; ``gc.freeze()`` moves it to the permanent generation so cyclic
    collections stop re-scanning it, and a larger gen-0 threshold makes collections rarer. This
    removes generation-2 pauses from request tails without disabling the collector.
    """
    gc.collect()
    gc.freeze()
    g0, g1, g2 = gc.get_threshold()
    gc.set_threshold(max(g0, 50_000), g1, g2)
