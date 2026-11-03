import numpy as np
import pytest

from wonhee.art import compose


@pytest.mark.parametrize("name", ["proj-filedex-treemap", "proj-filedex-embedding"])
def test_filedex_effects(name):
    kw = dict(size=(300, 200), palette="ember")
    a = compose.render_effect(name, seed=1, **kw)
    b = compose.render_effect(name, seed=1, **kw)
    assert a.shape == (200, 300, 3) and np.array_equal(a, b) and a.std() > 0.03
