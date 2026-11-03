import numpy as np
from wonhee.art import compose

def test_pixelpipe_deterministic():
    d = {"nodes": [{"id": c} for c in "abcd"], "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "a", "target": "d"}]}
    a = compose.render_effect("proj-awork-pixelpipe", size=(300, 200), seed=1, palette="teal-coral", data=d)
    b = compose.render_effect("proj-awork-pixelpipe", size=(300, 200), seed=1, palette="teal-coral", data=d)
    assert a.shape == (200, 300, 3) and np.array_equal(a, b) and a.std() > 0.02
