import numpy as np
from wonhee.art import compose


def test_trainnet_review_deterministic_with_and_without_data():
    data = [{"state": s, "x": x, "y": y} for s, x, y in
            [("confirmed", .2, .3), ("recovered", .7, .2), ("reshaped", .5, .6), ("dropped", .8, .8), ("deferred", .1, .9)]]
    a = compose.render_effect("proj-trainnet-review", size=(300, 200), seed=1, data=data)
    b = compose.render_effect("proj-trainnet-review", size=(300, 200), seed=1, data=data)
    c = compose.render_effect("proj-trainnet-review", size=(300, 200), seed=1)
    assert a.shape == (200, 300, 3) and np.array_equal(a, b) and c.std() > 0.02
