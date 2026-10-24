# Moeka and wonhee integration build plan

Status: approved 2026-10-24; owner requested implementation. Governed by the [integration spec](../specs/2026-10-24-moeka-integration-design.md).
This is an implementation sequence, not a claim that the workflow is available. Owner agreement on the new
contracts precedes implementation under [PROCESS.md](../PROCESS.md). Preserve existing Telegram identity and
the paired wonny workspace; do not mix this work with super-moeka/RSI changes.

## Decisions to settle from the proposal

The proposed defaults are: existing Telegram bot as the only poller; wonny as the dedicated wonhee assistant;
public single-file Drive links first; raw Telegram files within hosted bot limits; canonical creative memory
in wonhee; no automatic publishing. The current tailnet-only inbox design remains valid for private large
uploads. Telegram/Drive are additional intake routes, not a public upload server.

Agree on the creative-project and memory contracts and the extension to the peer envelope. A local Telegram
Bot API server can be decided later; it is unnecessary for the initial Drive-supported workflow.

## Wave 1 Establish the tool and conversation boundary

1. Add a checked wonhee MCP launch configuration for wonny using the existing wonhee venv, absolute cwd and
   explicit data home. Start with discovery/status tools and validate both child startup and reconnect.
2. Add wonhee operating instructions to wonny and a narrow handoff instruction to moeka. Use its already
   configured peer identity; keep secrets out of skills and memory. Persist the active creative context and
   the Telegram chat/topic route instead of inferring it repeatedly from prompt history.
3. Extend peer source/destination correlation and session selection. Current `peer:<sender>` session behavior
   needs a project distinction before simultaneous creative projects can be supported.
4. Add health output for tool connection, peer pairing, retrieval backend, intake queue and delivery queue.
   Existing vector configuration contains legacy keys accepted as extras; report the parsed effective config.

Check: the owner can select wonhee from the existing bot, discover real reel styles, and return to general
conversation. Two project threads remain separate. No paid model/GPU execution is needed for tool discovery.

## Wave 2 Save assets and creative memory

1. Design additive migrations for creative projects, versioned briefs, intake receipts, project assets,
   run associations, ideas and scoped memory. Keep the existing execution `project` hierarchy intact.
2. Add repo helpers and service/CLI/MCP actions with typed receipts and status. Implement local intake first:
   streamed copy, probe, sha, atomic storage, provenance and duplicate handling. Do not require analysis to
   finish before an asset is safely saved.
3. Extend moeka's Telegram attachment metadata and durable receipts. Preserve captions, album order and source
   reply identity; handle incomplete albums and replayed updates. Transfer approved refs into wonhee without
   widening the agents' filesystem/shell permissions to the whole home directory.
4. Add wonhee memory write/search/correct/forget and idea lifecycle actions. Retrieve exact project state and
   FTS results before adding semantic retrieval. Exclude unconfirmed suggestions and superseded facts.
5. Feed confirmed decisions and feedback into the brief and retrieval context. Update wonny instructions to
   use these actions; a remembered fact must have a successful durable write before it is acknowledged.

Check: “save these takes and this idea” survives service restart, repeated uploads preserve source receipts,
and “remember less text” followed by a correction affects a new session without embeddings or Dream.

## Wave 3 Support public Drive files

1. Build the deterministic binary-file adapter with file-id/resource-key parsing, redirect validation,
   streaming byte limits, probe and shared intake receipts. Start without OAuth and without folder recursion.
2. Fixture-test confirmation pages, interrupted transfers, denial, quota, misleading MIME, unsafe redirects
   and returned HTML. Document the chosen anonymous endpoint's limitations; use an explicit API-key route
   if the live evidence requires it.
3. Add useful oversized-Telegram guidance and distinguish unsupported link types from temporary failure.
4. Record a live import from a representative public file larger than Telegram's hosted download limit.

Check: a pasted Drive link and a Telegram file produce equivalent stored asset refs, project associations
and analysis-ready inputs. No third-party media enters git.

## Wave 4 Make videos and return files

1. Add durable creative submissions around existing reel/chain orchestration, with logical idempotency keys,
   brief revision and request/run correlation. Use detached chains and job status; wrap synchronous `reel`
   orchestration so it can finish across tool timeout/disconnection. Do not build the full timeline engine
   as a prerequisite for the existing reel styles.
2. Run requested transcript/contact-sheet work and the selected recipe. Return useful progress from recorded
   state. Keep model turns out of status-poll loops and enforce existing budget/GPU policy.
3. Add output manifests and a durable completion-to-delivery bridge. Extend peer results with artifact refs;
   do not squeeze file manifests into the existing limited text result.
4. Stage typed files into scoped delivery storage, stream Telegram uploads, save sent message ids and retain
   the original. Add the approved private phone outbox for oversized originals; previews are separate outputs.
5. Add the full file checker as a distinct dependency for checked badges. Until then show actual existing
   checks and unknown full-check status. Address isolated-voice audio delay D23 before certifying sync.

Check: a supplied take becomes a real video in Telegram, an older output can be revised, a gateway restart
does not duplicate rendering, and transfer failure remains independently retryable.

## Wave 5 Harden memory and recovery

1. Ship hybrid semantic/keyword recall in v1; select CPU embeddings explicitly and expose unavailable/degraded state. The currently running moeka venv lacks the embedding
   dependency, so no existing vector-memory setting proves semantic recall works.
2. Exercise cross-project exclusions, contradictory feedback, forgotten records, source attribution and
   derived-index rebuilds. Back up wonhee DB/media and wonny memory with SQLite-safe backup handling.
3. Exercise worker loss, duplicate transport updates, restart during album receipt, crash after send but
   before acknowledgement, disk-full download and peer outage. Do not replay uncertain side effects blindly.
4. Run the spec's complete acceptance set with creator media. Record actual wall times, costs and failures;
   any performance claim needs a bench or gauge record. GPU checks acquire the shared lock.
5. Have a different agent or person align each implementation wave against its governing spec, resolve
   mismatches, then update the design status and user-facing instructions.

## Exploration changes and validation

On 2026-10-24, a focused moeka fix corrected the missing `return_session_key` column in peer result lookup.
Two regression cases first reproduced `IndexError`: saved requesting session and legacy null-session fallback.
All 12 tests in `tests/test_peers.py` and ruff for the affected source/test pass afterward. The paired-agent
guide now states the routing behavior. Both moeka services were restarted to load the fix. Their `/health`
endpoints returned `ready: true`; Telegram reconnected and wonny's WebSocket reported running. The general
gateway also logged stale WebSocket destinations, tracked in the spec's moeka issues. No test messages were sent.

Wonhee's real MCP stdio handshake and discovery calls passed in a temporary data home, with GPU visibility
disabled and no model calls. The server listed 28 tools at inspection time. This validates transport and
discovery only. No real Telegram upload, Drive import, render, memory-isolation acceptance or artifact
delivery was performed. No runtime MCP registration or creative-memory migration was applied.
