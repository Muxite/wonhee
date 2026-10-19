# Working in wonhee (for agents and humans)

The README covers what wonhee is and how to use it. This page tells you where code lives, how to change it safely, and
the house rules. Read this first, then follow the links for the part you are touching.

## Map

| Layer | Where | What it does |
|---|---|---|
| Entry points | `wonhee/cli.py`, `wonhee/mcp_server.py` | Thin. They parse args, open the DB, call `service`, and print. Every MCP tool has a CLI twin |
| Service | `wonhee/service.py` | One function per user-facing action (`audio`, `vibe`, `shots_search`, `node_run`, `chain_run`, …). They return text |
| Nodes | `wonhee/nodes/*.py`, registered in `nodes/registry.MODULES` | Typed steps: `Node(name, version, inputs, outputs, params, fn, doc, salt=, lane=)`. Chains wire them together. Results are cached by (node, version, inputs, params, salt) |
| Engines | `wonhee/edit/` (pixels, audio; `edit/viz/` draws pictures from data), `wonhee/analyze/` (speech, shots, embeddings), `wonhee/muksite.py` (portfolio upload) | Plain functions on files and arrays. They know nothing about the DB |
| Store | `wonhee/db/` (`migrations/NNNN_*.sql`, `repo/`, `queries.py`) | SQLite in WAL mode. Blobs are content-addressed under `data/media/` |
| Warm workers | `wonhee/worker.py` | Long-lived process per lane. It keeps models loaded and runs detached chains from the `job` table |
| Profiler | `gauge/` (+ `profiling/`) | Runs any job in a container and measures CPU, RAM, VRAM and the GPU trace. See `docs/profiling/GUIDE.md` |
| Docs | `docs/production/` (rules, numbers), `docs/profiling/` (gauge, dataset), `docs/specs`, `docs/plans` | Every number in them was measured on this box |

## Commands

```bash
uv sync --all-extras --group gpu    # the gpu group (torch, transformers, demucs) is optional: masks, isolation, shot search
uv run pytest                       # default: no network, no GPU (markers `live` and `gpu` are deselected)
uv run pytest -m gpu                # real CUDA models. Hold the GPU lock (below)
uv run wonhee node list             # every node with its ports
uv run wonhee worker status         # warm workers and the job queue
uv run gauge doctor                 # profiler prerequisites (add --mps for the MPS daemon)
```

## Adding a node (checklist)

1. Write the engine function in `edit/` or `analyze/`, with no DB access, so it is easy to test.
2. Write the node in `nodes/<area>.py`:
   - It takes a params model with `extra="forbid"`.
   - It returns `{port: ref}`.
   - It calls `ctx.flag(CODE, msg)` for anything a reviewer should see.
3. `registry.register(Node(...))`:
   - Bump `version` whenever the output changes.
   - Put model names in `salt`.
   - Set `lane="cuda"` if it loads a GPU model. A live worker can then run it warm.
4. Append the module to `nodes/registry.MODULES` if it is new.
5. New flag codes or tables go in a new `db/migrations/NNNN_name.sql`. Never edit an applied migration.
6. If people or agents call it directly, add a `service` function plus its CLI and MCP twins.
7. Tests in `tests/test_<area>.py`:
   - Use the `conn`, `settings` and `tiny_video` fixtures (`tests/conftest.py`).
   - Use fakes from `tests/fakes.py`. Monkeypatch model modules (see `tests/test_shots.py`).
   - Anything that needs real CUDA gets `@pytest.mark.gpu`.
8. Update the README or `docs/production/` if behaviour or a rule changed.

## The GPU is shared: take the lock

Other agent sessions on this machine run timed GPU sweeps. Any GPU use counts, even a 5-second host smoke test. Before it:

```bash
/home/muk/projects/gpu-lock status                       # who holds it
/home/muk/projects/gpu-lock acquire "wonhee: <what>" --wait --ttl 1800   # prints a token
/home/muk/projects/gpu-lock release --token=<token>
/home/muk/projects/gpu-lock chat "wonhee: queued behind you"             # coordinate
```

`gauge run|sweep|corun` take the lock themselves. Never kill another session's GPU process.

## House rules

- **Measure before you claim.** Performance statements in docs cite a gauge run or a bench script. New GPU work gets a `wonhee.gpu` bench task (`wonhee/bench.py`) and a row in the experiment files under `profiling/sweeps/x/`.
- **Match the surrounding code:**
  - Module docstrings say why the module exists.
  - Comments explain decisions, not syntax.
  - Lines up to about 120 characters.
  - No new dependencies without a reason in the commit.
- **Commits:** a short lowercase summary in the existing style (`add …`, `fix …`), one topic per commit. Run `uv run pytest` before committing.
- **Data stays out of git.** `data/` (DB, media, models, gauge ledger and traces) is gitignored. Third-party media never goes in the repo.
- **Unknowns are `None`, never 0.** This holds both in gauge summaries and in node reports.

## Where to look when …

| You want to … | Read |
|---|---|
| know what a run did, or why it failed | `wonhee show run <id>` / `wonhee chain show <id>`, then README › Debugging |
| decide what runs where, or how many at once | `docs/production/policies.md` |
| pick the next thing to build | `docs/production/workers.md` › Build next |
| profile something | `docs/profiling/GUIDE.md`; experiments in `profiling/sweeps/x/` (run order: `profiling/sweeps/x/round2.sh`) |
| understand the GPU numbers | `docs/profiling/DATA_CARD.md`, `profiling/report/shared-gpu-node.html` |
