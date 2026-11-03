# Bridge contract with moeka

The trusted transport bridge (moeka, `nanobot/creative_bridge.py`) calls `wonhee.cli creative <action>` and
reads fixed keys from the responses. This file is the wonhee side of that agreement.

- `contract` (host-only, never an agent tool) returns `contract_version`, `revision`, `actions`, `shapes` and
  `signal`.
- `BRIDGE_SHAPES` in `wonhee/creative.py` lists the response keys the bridge reads per action.
  `tests/test_creative.py::test_bridge_shapes_match_the_real_responses` checks each against the real response.
- Adding keys or actions is compatible. Removing or renaming an action or a listed key bumps
  `CONTRACT_VERSION`; moeka then reports a mismatch at startup and in `/wonhee status` instead of failing quietly.
- `signal` names a marker file in `WONHEE_HOME`. `wonhee/db/repo/creative.py:signal_bridge` touches it when a
  delivery or submission becomes queued, failed or uncertain (and on a new delivery). Progress does not touch it.
  Long-lived processes (`creative serve`, the stagecraft worker) must be restarted to pick this up.
- `health` reports `stuck_deliveries` (sending, uncertain, failed) and `active_jobs` (queued or running, with
  `idle_s`, `stalled` and the delivery `route`).
- `delivery_show` accepts `uncertain` to `sent` when message ids are given: recovery marks every in-flight send
  uncertain, even one a live bridge completes afterwards.
