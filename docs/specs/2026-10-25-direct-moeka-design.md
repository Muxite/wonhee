# One conversational agent, Wonhee tools, Stagecraft jobs

Approved by owner 2026-10-25: remove Wonny from the active path and fix the observed
empty-spec retry loop, project context and reference access in that order.

1. Add CLI/MCP stagecraft_build with ordinary typed fields: brief, title, duration,
   surface, orientation, primitive and reference_asset_ids. The server builds the
   ProductionSpec, saves the canonical brief and calls the existing idempotent queue.
   Do not expose the legacy arbitrary-dictionary submit tool to the active agent.
   Project/request/recipient come from authenticated transport, not model arguments.
2. Expected MCP failures return CallToolResult(isError=true) with structured code,
   message and retryable=false, bypassing success-output validation. Moeka stops a
   repeated permanent tool failure after two identical calls; schema failures have
   a small per-turn ceiling. No unbounded same-input retry loop.
3. Moeka retains Telegram conversation, selected project and pending brief. An
   explicit direct bridge mode imports media and attaches host-authenticated creative
   context to the original Telegram turn, without a peer enqueue or second LLM.
   Project selection in the same turn updates that context and saves the initiating
   brief. Project snapshots include canonical brief or source-linked saved brief.
   Existing transport receipts/delivery acknowledgements remain authoritative.
4. Still-image media_info returns bounded image pixels through MCP. Reference asset
   IDs are checked against the project, copied as bounded image files, and passed to
   Codex via --image in its isolated authorship filesystem. No protected-directory
   bypass or broad workspace grant. Generated execution remains offline/containerized.

Deployment snapshots both agent configs/instructions, gateway receipt DB and Wonhee
DB. Configure Wonhee MCP on main Moeka, disable the legacy submit tool, enable direct
mode, stop/disable Wonny and preserve all of its files/history. Save the creator's
existing project2 brief/reference and make it available in the direct conversation.
Do not automatically replay uncertain Telegram sends or duplicate production jobs.
The obsolete peer mode remains compatible for existing checkouts/tests but is not
deployed on this host. Update Stagecraft's installer so rerunning cannot restore Wonny.

Acceptance: typed model-visible parameters; invalid request is a permanent MCP error;
duplicate failure stops the runner; direct Telegram context survives same-turn project
selection and restart; tampered metadata is refused; still image pixels reach MCP and
Codex; model-authored brief produces a real queued job with the original route. Exercise
the running main Moeka model through a captured outbound test sink, then real isolated
coding/capture/outbox. Independent alignment review before final commit/deployment.
