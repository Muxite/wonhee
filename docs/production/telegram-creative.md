# Telegram creative workspace

The existing `@moeka1738_bot` can hand wonhee work to its paired wonny instance. The host keeps one Telegram
poller, a separate creative conversation per project, canonical SQLite memory and content-addressed media.
The integration is implemented and independently reviewed; creator-file live acceptance is pending.

## Use it

1. Say you want to work on wonhee, or send `/wonhee project <name>` to select a project explicitly.
2. Send an idea, photo/video/document, album, or public Google Drive file link. Uploads are saved before analysis;
   a bare upload does not authorize rendering. Drive folders, private Drive access and native Docs/Sheets are
   outside v1. Hosted Telegram cannot download raw takes over 20 MB; use a public Drive file link for those.
3. Ask for an edit, for example “make this into a legend reel with the title campus startup”. The host queues
   and records its speech/render phases, then delivers the finished video. A follow-up request creates a revision.
4. Ask for the original as a document if wanted. Oversized originals have a private download link on the outbox:
   `http://100.66.112.22:17420`. Tailscale must be connected on the phone. A preview is distinct from the original.
5. `/wonhee leave` returns to general moeka. `/wonhee status` shows transport queue counts; `/wonhee retry`
   explicitly retries the latest failed/uncertain text reply. An uncertain send may already have reached Telegram.

The existing styles are `legend`, `explainer`, `scrapbook`, `series`, `list` and `lecture`. The dedicated assistant
can inspect real timestamped contact sheets with `media_info(analyze=true)` rather than guessing from a filename.
Full rendered-file checking remains unbuilt; “ready” means rendered, not certified against all posting checks.

## Memory in ordinary words

Stated preferences, decisions and feedback are saved automatically with their source. Corrections supersede
old records; forgetting removes a fact from normal recall and derived indexes. Tentative ideas remain distinct
from confirmed instructions. Examples:

- “Remember that my own clips should have understated captions.”
- “Actually, keep the words bigger on campus clips.”
- “What did I dislike about the last edit?”
- “Find the ideas involving campus footage that we haven't made.”
- “Show the decisions for this project in order.”

Search combines SQLite FTS and CPU sentence embeddings using the cached `all-MiniLM-L6-v2` model. Ordinary-word
queries can also select a project, type, status or date range, count records, or read a timeline. Cross-project
search stays within wonhee; unrelated general moeka memory is excluded. If semantic retrieval fails, keyword and
structured queries continue and health reports the error. Exports live under `data/creative-memory/`:
`MEMORY.md` for reading and `records.json` for records, sources, revisions and tombstones. SQLite is authoritative.

## Operate and recover

```bash
uv sync --all-extras --group gpu
uv run --no-sync wonhee creative health
uv run --no-sync wonhee creative make_show --data '{"submission_id":1}'
uv run --no-sync wonhee creative delivery_show --data '{"delivery_id":1}'
uv run --no-sync wonhee creative memory_query --data '{"query":"understated captions","scope":"all"}'
systemctl --user status wonhee-creative moeka moeka@wonny
journalctl --user -u wonhee-creative -u moeka -u moeka@wonny -n 60
```

All creative actions have `wonhee creative <action> --data '<JSON>'` and matching MCP tools. `project_*` actions
use the `creative_project_*` MCP names. MCP `delivery_show` is read-only: only the host writes transport
acknowledgements. The bridge validates delivery routes against actual Telegram ingress; agent-chosen recipients
cannot redirect a result. It only reads staged outputs under `data/creative-outbox/`.

`make_submit` takes `project_id`, `asset_id`, a stable `request_key`, style, optional brief, trusted route and
`params` (`title`, `cta`, `items`, `slots`, `tighten`, `look`, `quality`, `cost_cap_usd`). Reuse the request key to
reconcile a timeout; a different brief/options under the same key is refused. A new revision gets a new key.
`make_show` returns exact phase/run associations. Do not repeatedly ask an LLM to poll it.

A worker restart reconciles recorded local phases; a gateway restart recovers durable input and output queues.
In-flight sends become uncertain, so they are not blindly replayed. A failed transfer does not make a render
failed. Use `delivery_retry` only when the creator asks to resend a failed/uncertain artifact. Intake retries
serialize by source key, so a timed-out caller cannot start a second download of the same receipt.

Install/update the local instance configuration with moeka's interpreter:

```bash
/home/muk/projects/moeka/.venv/bin/python scripts/install_creative_bridge.py       # validate/dry run
/home/muk/projects/moeka/.venv/bin/python scripts/install_creative_bridge.py --apply
systemctl --user daemon-reload
systemctl --user enable --now wonhee-creative.service
systemctl --user restart moeka@wonny.service moeka.service
```

The installer snapshots the four changed instance files and backs up the live wonhee DB with SQLite's backup
API under `data/deployments/<UTC time>/`. It never copies bot/provider secrets between instances or prints them.
The private outbox binds to the Tailscale IPv4 address by default. For another host, adjust the example unit's
address; no public tunnel or public storage is used. Config and service snapshots can restore the old runtime;
leave additive migrations in place rather than deleting live rows.

The creative daemon takes the shared GPU lock for real submission work and releases its transcriber before
releasing the lease. Existing warm-worker/GPU policies still apply. CPU search and contact sheets need no GPU.

## Validation record

2026-10-24: baseline 821 tests passed. The integration's full suite reached 868 passing tests; subsequent
focused checks validate real CPU contact sheets/preview encodes, MCP images, scope filtering, concurrent
receipt/index writers, original document mode, and range downloads. Independent reviewers checked both sides
of the bridge. Real CPU semantic retrieval matched “understated captions” against a differently worded saved
preference. These are local results; they do not stand in for Telegram/Drive live transfer acceptance.

Deployment: all three services report ready; Telegram reconnected and wonny registered 26 allowed wonhee tools.
Rollback snapshot: `data/deployments/20261024T041130Z/`. Creator test files and live phone-download verification
remain outstanding. The complete file checker and isolated-voice audio-delay fix D23 remain separate work.
