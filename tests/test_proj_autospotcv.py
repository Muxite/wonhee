import numpy as np
from wonhee.art import compose, metrics


def test_embedding_deterministic_and_all_over():
    a = compose.render_effect("proj-autospotcv-embedding", seed=1, size=(600, 400), palette="violet-dusk")
    b = compose.render_effect("proj-autospotcv-embedding", seed=1, size=(600, 400), palette="violet-dusk")
    assert a.shape == (400, 600, 3) and np.array_equal(a, b) and a.std() > 0.03
    rep = metrics.analyze(a)
    assert rep.center_bias <= 1.6 and rep.coverage >= 0.75
