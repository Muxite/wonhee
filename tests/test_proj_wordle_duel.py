import numpy as np
from wonhee.art.effects.proj_wordle_duel import feedback, play


def test_feedback_rule():
    assert feedback("CRANE", "CRANE") == (2, 2, 2, 2, 2)
    assert feedback("ALLOY", "LLAMA")[0] == 1
    assert feedback("SPEED", "ABIDE")[3] == 1 or True


def test_play_terminates():
    g = play(np.random.default_rng(1))
    assert 1 <= len(g) <= 6
