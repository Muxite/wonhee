"""Check every registered effect's default render contract and seeded determinism."""

import re
from time import perf_counter

import numpy as np
import pytest

from wonhee.art.compose import render_effect
from wonhee.art.features import synthetic_source
from wonhee.art.registry import discover, names


discover()


@pytest.mark.parametrize("name", names())
def test_default_render(name):
    source = synthetic_source((96, 64), seed=0)
    started = perf_counter()
    try:
        try:
            first = render_effect(name, size=(96, 64), seed=0, palette="ember", source=source.copy(), data=None)
        except Exception as exc:
            # Only explicit missing-input errors qualify; unrelated render bugs must fail.
            message = str(exc)
            if re.search(r"\b(data|graph|matrix|points)\b", message, re.IGNORECASE) and re.search(
                r"\b(require[sd]?|missing|must|provide[ds]?|expected|none)\b", message, re.IGNORECASE
            ):
                pytest.xfail(reason=message)
            raise

        assert first.dtype == np.float32
        assert first.shape == (64, 96, 3)
        assert np.isfinite(first).all()
        assert (first >= 0).all() and (first <= 1).all()

        second = render_effect(name, size=(96, 64), seed=0, palette="ember", source=source.copy(), data=None)
        np.testing.assert_array_equal(first, second)
    finally:
        elapsed = perf_counter() - started
        assert elapsed <= 20, f"{name} took {elapsed:.3f}s (limit: 20s)"
