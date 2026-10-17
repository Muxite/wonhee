"""gauge: run a declared job in Docker, freeze a prediction first, measure from the host, score the prediction.

Library surface for other projects (wonhee preflight, moeka launchers):
    from gauge import estimate, preflight
"""
from gauge.preflight import estimate, preflight  # noqa: F401
