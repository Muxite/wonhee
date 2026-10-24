# Moeka and wonhee integration

Status: approved 2026-10-24 by the owner’s instruction to implement the final plan. Large lane under
[PROCESS.md](../PROCESS.md). This extends stages 0, 2, 3 and 6 without changing
[VISION.md](../VISION.md). Implementation sequence: [build plan](../plans/2026-10-24-moeka-integration.md).

The creator should be able to send a Telegram message, take, reference or public Drive file, discuss an idea,
receive an edit, and ask for changes days later. Reuse the existing Telegram moeka and its paired `wonny`
instance. Give wonny responsibility for wonhee work, with durable project records in wonhee and separate
conversation memory in wonny. Keep one Telegram poller and use wonhee's existing service and MCP boundaries.

## Baseline before implementation

Inspection on 2026-10-24: wonhee checkout `ca9652a`; moeka checkout `671421aa` before the peer fix below.

| Component | Observed state | Implication |
|---|---|---|
| Telegram | `moeka.service`, workspace `~/.nanobot`; Telegram enabled for the configured owner | Keep this as the transport and general assistant |
| Wonny | `moeka@wonny.service`, workspace `~/.moeka-wonny`; Telegram disabled | Separate sessions, memory and credentials already exist |
| Pairing | Both instances configured reciprocally; authenticated Unix peer sockets exist | Extend this bridge rather than add another agent framework |
| Wonhee access | Neither instance has MCP servers configured | The conversational agent cannot currently discover wonhee tools through MCP |
| Wonhee MCP | Real stdio handshake, tool listing, `nodes_list(compose_video)` and `reel_styles` succeeded in a temporary data home | Existing tool transport works; processing and delivery still need integration |
| Rendering | `reel` uses speech timing and `compose_video`; `chain_run(wait=False)` supports detached work | Build on these paths; the general timeline renderer remains unbuilt |
| Media | Local paths can enter chains and become content-addressed refs | Storage exists; intake receipts and source associations do not |
| Memory | Moeka has explicit facts, Dream, history and scoped document stores | Reuse retrieval, but add canonical creative records and project boundaries |
| Embeddings | `sentence_transformers` absent from the running moeka venv | Semantic recall is unavailable there; lexical recall must work independently |
| Checks | Full rendered-file checker and phone inbox/outbox are approved but unbuilt | Delivery must accurately distinguish checked and unchecked output |

Relevant code: `wonhee/service.py`, `wonhee/mcp_server.py`, `wonhee/worker.py`, `wonhee/db/repo/`;
moeka's `nanobot/peers.py`, `agent/tools/peer.py`, `agent/tools/memory.py`, `kernel/memory.py`,
`channels/telegram/runtime.py`, `config/schema.py` and `docs/guides/paired-agents.md`.

## Conversation and ownership

```text
Telegram messages and attachments
             |
             v
moeka gateway: owner identity, chat/topic route, durable receipt
             |
             | authenticated peer request with creative project and source refs
             v
wonny: wonhee conversation, planning, selected creative memory
             |
             | wonhee MCP, short calls and detached jobs
             v
wonhee: intake, projects, memories, media, runs, checks, outputs
             |
             | durable completion and artifact refs
             v
moeka gateway: Telegram preview and original file or private outbox link
```

Moeka owns chat transport, paired identity, source-message receipts and delivery attempts. Wonny owns the
creative conversation and chooses wonhee actions. Wonhee owns media, ideas, confirmed creative preferences,
brief revisions, runs, outputs and their provenance. Wonny's summaries and retrieval index are rebuildable
views of creative records; Dream cannot become the sole source of an accepted brief or remembered correction.

Use the existing paired instance before considering an embedded `AgentSpec`. The kernel already supports
`memory_key` and document scopes, but moving the running gateway onto a different hosting model adds work
without solving file delivery. Wonny's `SOUL.md` currently describes tone, not a wonhee role; configure that
role through its operating instructions and selected tools, preserving personality.

Proposed interactions:

- “Let's work on wonhee” selects a durable creative context for this Telegram chat/topic. Subsequent files and
  ideas belong there until the creator switches or leaves it. Explicit `/wonhee` is an optional recovery path.
- “Use these for the midnight water reel” creates or selects a named creative project and records its assets.
- A file without an instruction gets a receipt and is saved to the current project or an unassigned inbox.
  Do not render just because an upload arrived. Ambiguous raw-footage versus inspiration roles get one question.
- “Make it shorter, like the last one” resolves the replied-to output first, then the active project and
  recorded preferences. Ask only if the target or requested edit remains ambiguous.
- New topics and projects get separate conversational sessions. A topic session alone is insufficient for
  memory isolation: current explicit facts are workspace-wide, and peer ingress uses one session per sender.

Persist routing by owner, channel, chat and topic. Carry source message ids, creative project id, brief
revision and request id through the peer bridge. Do not infer authority or a reply destination from message
text. The Telegram bot token stays only with moeka; wonny never starts polling it.

## Intake

### Telegram files

Persist a receipt before asynchronous work, keyed by bot instance, chat id and message id plus attachment
ordinal. Preserve album id, order, sender, caption, reply id, original name, MIME, byte count, `file_id` and
`file_unique_id`. Keep retryable transport references separate from stable media hashes.

Download into a temporary file, enforce configured byte and disk limits, probe actual content, hash it, then
atomically store it. A completed receipt returns an asset id, sha, media type and probe summary. Replayed
updates reuse the receipt; identical bytes reuse storage but retain each source association. Failed or partial
downloads cannot appear as ready assets. Album completeness must survive slow downloads and gateway restarts.

Save first; run transcript, contact-sheet and shot indexing as explicit jobs with independent status. Preserve
raw files. Prefer sending photos/videos as documents when retaining the original matters; compressed Telegram
media remains usable but its received form is what wonhee stores.

Telegram's hosted API currently limits bot downloads to 20 MB and multipart video/document uploads to 50 MB.
A local Bot API server supports downloads without that size limit and uploads up to 2000 MB.
[Telegram API](https://core.telegram.org/bots/api#getfile),
[local server](https://core.telegram.org/bots/api#using-a-local-bot-api-server).
For the first release, detect oversized Telegram input and suggest the supported public Drive path or private
phone upload. Add a local Bot API server only if direct large Telegram transfers are needed; moeka must first
gain explicit base URL, file base URL and local-mode configuration. Do not promise that sending “as a file”
bypasses the hosted bot download limit.

### Public Google Drive links

Drive is input only; outputs use Telegram and a private Tailscale download page. Accept canonical binary-file
links and `open?id=...`, preserving `resourcekey`.

Folder extension (built, independently aligned 2026-10-24; owner request): accept public `/drive/folders/ID` and
`/drive/u/0/folders/ID` links. Enumerate the anonymous embedded folder view recursively, preserving resource
keys on child links. Reject permission/login pages, native Workspace documents, shortcuts, unsafe names,
duplicate sibling names and cycles rather than silently omitting files. Bound traversal to 1000 entries,
32 folder levels and 8 MiB per listing. If Google exposes pagination, fail rather than claim completeness.
No OAuth, API key, browser cookies or new dependency is required.

`drive_download` is a CPU, non-cacheable node: no input ports; params `url`, `dest`; output `manifest:file`
(JSON containing destination and ordered files with relative path, local path, Drive id, bytes, MIME and stored
ref). `wonhee download URL DEST` and MCP `drive_download(url, dest)` call the same service. DEST is a folder;
create it when missing, retain the folder's child hierarchy, and refuse existing file collisions. Validate the
entire listing and all output paths before transferring bytes. Stream each file through the existing bounded
binary adapter into a temporary file, then publish without overwriting; failed partial files are removed,
completed files remain available and a failure is reported. Apply the configured intake byte limit to the
whole transfer and keep the disk reserve. An individual file link also downloads into DEST using its filename.
MCP destinations remain confined to the existing output roots, including node_run/chain_run params.

`inbox_add` also accepts folder links and optional `dest` (default `<home>/drive/<folder-id>`); it downloads
and registers each child as a separate asset in the selected project, returning `assets` and `destination`.
A parent receipt in `creative_intake` shares the source-key uniqueness domain with file intake and checkpoints
the downloaded manifest before registering children. Stable child receipt keys derived from the request,
Drive id and relative path make successful replays idempotent. `inbox_list` accepts `limit` (default 200,
1–1000) and `offset` (default 0) so large imports remain discoverable. The Telegram bridge forwards at most
32 assets in a peer context and tells the assistant to page through inbox_list for larger imports. Files
retain folder-relative names in metadata. No analysis or GPU work starts on import. Single-file return shape
is unchanged. Folders with no binary files return an empty assets list. Shortcuts, native Docs/Sheets/Slides,
Google Vids and private Drive OAuth remain unsupported.

Use a deterministic Drive adapter in wonhee, behind the same intake service used for local files. Resolve the
file identity and download response, stream bounded bytes, validate every redirect and resolved destination,
then probe and store exactly as for Telegram. Browser confirmation forms are untrusted data. Reject login,
permission, quota and HTML interstitial responses with actionable errors; never ingest a login page as a video.
Keep resource keys and temporary download URLs out of ordinary logs and generated captions.

Google documents separate download methods for binary files and native Workspace documents, and resource keys
for some shared links. These docs do not guarantee a stable anonymous download endpoint for every public URL.
Choose and fixture-test the anonymous adapter against representative real links before declaring support.
An API-key-based public-file implementation is an explicit alternative if anonymous downloading proves
unreliable; it must not silently require personal OAuth.
[Downloads](https://developers.google.com/workspace/drive/api/guides/manage-downloads),
[resource keys](https://developers.google.com/workspace/drive/api/guides/resource-keys).

Suggested states: `received → downloading → stored`, with `failed` and an explicit retry. Analysis status is
separate from intake status. Persist resumable checkpoints only where the remote server supports safe range
requests and a stable version/validator; otherwise retry the download into a new temporary file.

## Creative projects and memory

Use a creative project above the existing execution `project → run → step` hierarchy. Today every chain
creates an execution project; a creator's reel must span many such runs, revisions and source messages.
Do not repurpose a chain's title or free-text notes as this relationship.

Proposed additive tables, with names finalized in the migration review:

| Record | Required information |
|---|---|
| `creative_project` | id, owner, name, lifecycle state, timestamps |
| `creative_brief` | project, revision, structured brief, source message, accepted status |
| `intake` | idempotency key, source identity, download state/error, original metadata, nullable media ref |
| `creative_asset` | project, intake, media ref, role: raw/reference/audio/output, source attribution |
| `creative_run` | creative project, execution run, brief revision, logical request id |
| `creative_idea` | project, idea text/ref, source citations, lifecycle and revision |
| `creative_memory` | scope, type, content, source event, time, status, superseded id |
| `creative_delivery` | output ref, run/step, requested recipient route, transfer state, sent message ids |

Memory scopes are creator-wide `wonhee` preferences and a specific creative project. Types include preference,
constraint, decision, feedback and open question. Draft model suggestions remain proposals; user acceptance or
correction determines a confirmed record. Keep contradictions and revisions traceable, exclude superseded facts
from normal recall, and distinguish forgetting a preference from deleting its underlying media.

Before planning or rendering, retrieve: creator constraints from `creator.yaml`, accepted brief, active project
assets, unresolved decisions, recent outputs and relevant confirmed memories. Ship exact lookup, FTS and CPU semantic retrieval in v1;
semantic retrieval is a rebuildable index using the cached all-MiniLM-L6-v2 model. Return source ids and scope with every recalled fact. Missing
memory is reported as missing, rather than invented from a nearby project. Unrelated general moeka memory is
excluded. Selectively import only creator-confirmed wonhee facts from older conversations.

Persist a correction such as “less aggressive captions on my own clips” before acknowledging that it is saved.
The next session must retrieve it even if Dream has not run. Wonny can keep short conversational continuity in
its existing memory, but canonical creative facts go through wonhee's memory service. This is separate from
the approved style-memory R6 feature, which measures voice/rhythm from chosen posts and applies a no-copy guard.

## Execution and delivery

Wonny uses wonhee MCP launched with absolute interpreter, module, working directory and data-home paths. Keep
wonhee's pinned kernel dependency and moeka's runtime in separate environments. Load wonhee keys from wonhee's
existing project configuration; do not copy Telegram or unrelated moeka secrets into the child.

Start with a small allowlist of required tools. Discovery/status are `nodes_list`, `reel_styles`, `workers`,
`chain_show`; production adds the approved intake, memory and submission actions. `publish_muksite` must not
be exposed until there is an actual approval mechanism for it in this Telegram route. Its current MCP contract
relies on a client tool-approval prompt, which an unattended Telegram call cannot assume exists. Preserve scout's
explicit “go” gate. Bind any public-action approval to the exact artifact, destination and operation.

All new user actions have service, CLI and MCP twins. Candidate actions are `inbox_add`, `inbox_list`,
`media_info`, `creative_project_*`, `idea_*`, `memory_*`, `make_submit`, `make_show`, `outbox_add`, `outbox_list`.
These are proposed names, not tools available today. File transformations and analysis remain typed nodes;
state and routing actions follow the existing service/store ownership rules. Every DB write uses a repo helper.

Return typed status and refs for new integration actions while preserving existing human text surfaces. Avoid
parsing a paragraph to recover run ids or file paths. Reads, writes and paid/GPU execution have separate effects.
Create a durable submission keyed by logical request and brief revision before launching work, closing the
crash window between starting a run and recording its correlation.

Use `chain_run(wait=False)` for suitable chains. Current `reel` orchestration is synchronous and spans several
runs; wrap the whole workflow as a durable submission before exposing it as a background action. Merely raising
moeka's default 30-second MCP timeout does not make that workflow restart-safe. A timeout yields unknown
execution state until reconciled by request id; do not automatically start the same render again.

The bridge watches persisted state and schedules completion delivery without keeping an LLM turn polling.
Keep routing metadata and output manifests across moeka and wonny restarts. Current peer messages/results carry
text, with a 12,000-character limit; extend their versioned envelope with validated artifact refs and project
correlation. A raw local path in peer prose is insufficient: both instances are workspace-restricted.

Grant only the intake/export directories or exact refs required by the transfer adapter. Use a scoped staging
copy for files, with an extension and MIME derived from validated metadata: the content-addressed store paths
have no useful extension for Telegram's media-type detection. Deliver a preview plus the original as a document
when fidelity matters; a preview does not replace the final artifact. Stream file handles instead of reading
the whole output into RAM. Oversized outputs retain a private tailnet outbox link when available. Do not
publish an output just to make delivery easier; the tailnet phone outbox is still an implementation dependency.

Delivery acknowledgements are separate from render completion. Retry bounded failures and record Telegram
message ids. A successful enqueue to moeka's in-memory bus is not a Telegram delivery acknowledgement. After a
crash between sending and recording success, delivery can be uncertain: avoid claiming exactly-once sending,
and provide a controlled resend. Returning files to the requesting owner is part of the creation request.

Keep paid work inside the existing run cap, report pre-run conversational/intent costs separately, and link
both ledgers to the creative request. GPU analysis and rendering obey the shared GPU lock and worker policies.
Full file-check badges stay unknown until the real checker has run. Preserve known audio sync drift D23 as a
separate fix; do not call an isolated-voice render fully checked while it remains open.

## Moeka issues and changes

| Priority | Finding | Required change |
|---|---|---|
| Fixed | Nested peer results read `return_session_key` without selecting it from SQLite | Query corrected; saved-session and legacy-fallback regression cases added; 12 peer tests and ruff pass |
| P0 | Every request from one peer enters `peer:<sender>` | Add trusted creative-project correlation and session selection; concurrent projects cannot share a creative thread |
| P0 | Peer requests/results carry text, not validated media/artifact envelopes | Extend existing durable transport with versioned refs, source metadata and destination correlation |
| P0 | General instance memory is broad; explicit facts use one workspace file | Keep wonhee conversations in wonny and use wonhee's scoped canonical creative records |
| P0 | MCP has a 30-second default timeout; `reel` is synchronous | Durable workflow submission and reconciliation; short tool calls return request/run ids |
| P0 | Peer results mark published after enqueueing to an in-memory bus | Persist a channel delivery job and acknowledge actual transport success |
| P1 | Telegram downloads preserve a path but discard original metadata in the bus payload | Add a typed attachment descriptor without breaking existing `media` consumers |
| P1 | Album buffers and ingress reorder queues are in memory; stop clears them | Durable receipts and album membership before acknowledging work; recover partial albums |
| P1 | Album flush starts after the first download with a fixed 0.6-second delay | Exercise slow subsequent downloads; accumulate by source album/order and allow late-member reconciliation |
| P1 | Telegram send uses `Path.read_bytes()` and reports generic transfer failures | Stream uploads; size preflight, actionable fallback and persisted delivery errors |
| P1 | Embedding code chooses SentenceTransformer's default device; dependency absent in live venv | Expose effective retrieval health; explicit CPU embeddings if enabled, lexical fallback always works |
| P1 | Hosted Telegram file limit has no dedicated fallback in this intake path | Size-aware failure points to Drive/private upload; optional local API configuration later |
| P1 | Restart emitted repeated `Unknown channel: websocket` warnings on the general gateway, where WebSocket is disabled | Audit persisted return routes and stale deliveries; surface undeliverable destinations without silent success |

The album and crash behaviors above are code-path findings, not measured loss rates. Do not redesign moeka's
agent loop for this integration: use channel attachment types, peer transport, tools and service adapters.
Keep this host's existing restriction, Bubblewrap sandbox and disabled sudo policy.

## Acceptance

1. Send text, a voice idea, a photo, a document and a small video; each gets one source receipt, correct role
   and project association. Model access to media uses real file refs/contact sheets, not a textual claim.
2. Send an album with captions and slow later downloads; preserve all members and order across restart.
   Replay updates; no duplicate logical intake or creation job. Same bytes retain multiple source associations.
3. Download a public Drive binary with and without a resource key, including one over the hosted Telegram
   download limit. Permission/quota/HTML failures remain actionable; unsafe redirects are refused.
4. Start a reel, end the LLM turn, restart each assistant separately, and receive the output for the same
   request. Stop the worker mid-job and reconcile existing runs before retrying. An MCP timeout cannot duplicate work.
5. Create two projects in parallel and reply to an older output; use the right source, revision and route.
   A completion arrives in the originating topic even after the active project has changed.
6. Remember a preference, correct it, restart, compact conversation and disable embeddings. Recall the corrected
   fact with its source; old and unrelated general-agent facts stay out. Forget removes it from derived indexes.
7. Deliver a playable preview and original document; an oversize final has a working private route. A failed
   send leaves the render successful and delivery retryable, with uncertain-send cases shown accurately.
8. Public actions require a concrete human approval; paid execution stays capped and GPU work takes the lock.
   Default operation does not add private Drive OAuth or post to social accounts.

Record live examples and observed failures for each wave. The stdio smoke and peer unit tests establish only
their respective boundaries; they do not establish an end-to-end Telegram video workflow.

## Approved implementation details

The owner selected seamless Telegram handoff, automatic retention of stated decisions/feedback, public Drive
imports, private tailnet downloads for oversized originals, and a request for live test files when ready.
Memory ships with CPU hybrid retrieval in v1, ordinary-word queries, source citations and text exports.

New service actions return JSON-compatible dictionaries (also structured MCP results); existing text actions
remain compatible. CLI twins use `wonhee creative ACTION --data JSON` for all new actions. The action names
are `project_create`, `project_select`, `project_show`, `inbox_add`, `inbox_list`, `media_info`, `idea_save`,
`idea_list`, `idea_update`, `memory_remember`, `memory_query`, `memory_correct`, `memory_forget`,
`memory_export`, `memory_reindex`, `make_submit`, `make_show`, `outbox_add`, `outbox_list`, `delivery_show`,
`delivery_retry`, and `health`. MCP prefixes project actions with `creative_`; other names match.

Downloads default to a 10 GiB per-file cap, retain at least 1 GiB free disk, and enforce timeouts. Memory
queries accept words plus optional project/type/status/date filters and search/list/count/timeline operations;
wonny translates the question into those typed arguments without another internal model call or raw SQL.
Creator-wide and project scope are explicit. Confirmed user statements are saved automatically; inferred
preferences remain unconfirmed. SQLite records are canonical, Markdown/JSON exports and semantic indexes are
rebuildable views; failed exports/indexing remain visible and retryable.

### Delivery preview node

`delivery_preview` v1 is a CPU node with `video → video`. Its forbidden-extra params are `max_seconds=60`,
`width=720`, and `max_bytes=45000000`. It creates a clearly labelled, bounded preview of an oversized original,
using x264/AAC and faststart. The original remains the authoritative output. If the bounded preview cannot be
produced, retain the original's private link and report preview failure without marking the render failed.

`contact_sheet` v1 is a CPU node (`video → image`) with params `frames=6` (1–12). It samples evenly spaced
real frames into a timestamped sheet; no model or GPU is used. `media_info(asset_id, analyze=true)` runs it
explicitly and returns its image/ref and run id. MCP additionally returns the image content so the assistant
can inspect footage without broad filesystem access. Probe-only intake does not start this analysis.

## Implementation evidence

2026-10-24: implementation and independent alignment are recorded in
[production instructions](../production/telegram-creative.md). Source/tests cover scoped hybrid memory,
source-preserving intake, per-project peer sessions, recorded execution phases and acknowledged delivery.
Both intake and derived-index writers serialize across processes. Contact sheets and previews are real CPU
media artifacts with node provenance. The first creator Drive video was imported and returned unchanged through
Telegram, with creator-confirmed playback and quality. Raw Telegram input, rendered/revised reels and private
phone-outbox acceptance remain pending, so the overall integration status remains approved.

### Live intake error correction

2026-10-24 live evidence: the creator supplied a public Drive folder. V1 correctly rejected its unsupported
shape, but the moeka CLI boundary hid the permanent error and retried it five times. This violates actionable
intake errors and prevents an otherwise valid creator message from reaching memory/conversation.
Creative CLI expected errors now use a bounded JSON error envelope on stdout with code/message/retryable and
nonzero exit. `invalid_request` is permanent; `transfer_failed` distinguishes transient network/server failures
from sharing, unsupported-link and size errors. Do not include URLs or structured request arguments in ordinary
error logs. Moeka forwards the creator's message and any successful assets even when a source is permanently
unsupported, with source warnings in signed `input_errors`; unsupported inputs are never reported as imported.

### Automatic memory provenance

Live evidence 2026-10-24: wonny saved the folder reference with an empty `source`, and initially proposed an
unsupported memory kind. Memory kinds/status/scope/operations must be enums in the MCP schema. For the
owner-configured local wonhee MCP connection, moeka forwards only the validated creative peer context in MCP
request metadata (`wonhee`), outside model-controlled arguments. Wonhee attaches its immutable request/message/
route identity to memory, idea and brief writes and binds delivery actions to the original destination. The
model can add domain citations but cannot override the transport identity. This opt-in is disabled for other
MCP connections and restricted to local stdio. Manual CLI calls remain explicit-source operations.

Within a forwarded creative conversation, `scope=auto` defaults memory writes/queries to the active project.
Use `memory_remember(scope=creator)` explicitly for wonhee-wide preferences; this is distinct from FastMCP's
filled `project_id=null` default. Manual CLI calls without transport context retain their existing null-project
behavior. Forgetting and idea-status changes preserve their original source plus the acting source in a nested
audit field. One creative creation request is keyed by its transport request id; retries cannot invent a new key.
