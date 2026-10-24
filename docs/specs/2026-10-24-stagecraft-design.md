# Stagecraft prop production and delivery

Status: approved by owner request to add the complete Stagecraft video delivery path
and containerize demo dependencies. Fits VISION's prop builder and capture/make/deliver
areas. This specification is committed before implementation.

Wonhee delegates technical-demo production to the sibling Stagecraft repository.
Stagecraft runs generated browser/terminal scenes in a shared, bounded offline Docker
image. Wonhee does not install demo dependencies or duplicate Stagecraft's renderer.

New CLI/MCP twins are `stagecraft_schema`, `stagecraft_submit(spec, project_id,
request_key, route)`, `stagecraft_show(job_id)` and `stagecraft_retry(job_id)`.
The schema tool documents the structured production spec and custom React component
contract. Submit validates the existing creative project and forwards to Stagecraft's
durable local queue. In a signed creative conversation, project, request key and route
are taken from validated transport context, overriding model-supplied identities.
Show/retry in creative context must remain within the source creative project.
Queued is not complete; render and delivery failure are reported separately.

Host-only CLI action `stagecraft_complete` imports an actual MP4 and creates one outbox
delivery for the original route, with a transactional receipt keyed by job ID. A replay
with changed project/route/content is refused. It is excluded from MCP discovery so
the agent cannot bypass trusted completion. Stagecraft's worker calls it only using
the persisted trusted job context. No new Telegram poller or publishing action.

The completion receipt links creative project, source request, media hash, delivery
and capture report. Stagecraft retains the source/demo/event/capture provenance;
Wonhee stores the original rendered bytes and references that manifest in intake
metadata. Output is labelled as staged/simulated unless an actual mechanism has been
implemented and separately verified. Existing transport acknowledgement determines
whether a staged file was actually sent.

Deployment snapshots wonny configuration/instructions, extends the existing tool
allowlist, and restarts only wonny to register tools. The existing restriction and
Bubblewrap settings remain. Test against temporary WONHEE_HOME before live migration.
Separate-agent alignment review is required by PROCESS.md before marking built.

## Implementation evidence

Built (aligned 2026-10-24). CLI/MCP twins delegate to the sibling Stagecraft queue;
completion is host-only and uses a transactional receipt. Source context overrides
model-supplied project/request/route. The full Wonhee suite passes (903 tests, 4
deselected). New schema/tool tests cover route binding, scope, private completion,
STRICT upgrade and delivery deduplication. The independent alignment review is clean.

Real isolated acceptance exercised MCP submission, offline container capture, media
intake and original-route outbox staging. The installed Stagecraft user worker also
consumed and completed a real queued test job. Wonny registered the four new allowed
tools without changing its filesystem restriction or Bubblewrap configuration.
Detailed artifacts and capture checks: sibling Stagecraft docs/VALIDATION.md.
No Telegram test was sent; creator phone playback is the first-request acceptance.

## Coding dispatch extension

Approved by the owner's request for a later agent-built stdio mesh demo. Existing
submit accepts a spec with optional coding {task, primitive: stdio_mesh}; Stagecraft
dispatches its local Codex builder, verifies real subprocess pipe traffic in an
offline container, binds the observed trace to a cinematic dashboard, captures and
stages via the same trusted route. No new Wonhee tools or provider credentials.
The schema/instructions expose this capability and coding/verifying status. Failed
verification cannot be delivered as successful real primitive work. See sibling
Stagecraft docs/CODING_DESIGN.md. Built (aligned 2026-10-24): actual systemd dispatch
used one Codex attempt, passed observed worker-pipe traffic and recorded/staged an
isolated smoke video. Stagecraft has 35 passing tests, including local-echo bypass
and unmatched-worker negative cases plus authored-bundle restart recovery. Wonhee's
full 903-test suite passes. The final themed creator request remains unsubmitted.
