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
existing project 2 brief/reference and make it available in the direct conversation.
Do not automatically replay uncertain Telegram sends or duplicate production jobs.
The obsolete peer mode remains compatible for existing checkouts/tests but is not
deployed on this host. Update Stagecraft's installer so rerunning cannot restore Wonny.

Acceptance: typed model-visible parameters; invalid request is a permanent MCP error;
duplicate failure stops the runner; direct Telegram context survives same-turn project
selection and restart; tampered metadata is refused; still image pixels reach MCP and
Codex; model-authored brief produces a real queued job with the original route. Exercise
the running main Moeka model through a captured outbound test sink, then real isolated
coding/capture/outbox. Independent alignment review before final commit/deployment.

Status: built and independently aligned 2026-10-25. Reviewer findings (multimodal
MCP errors, explicit switch versus replied-to project, hybrid portrait validation)
were fixed and regression-tested. Main Moeka is deployed in direct mode; Wonny is
disabled, with histories preserved. Original project 2 brief from confirmed memory 6
is now canonical and image 11 is unchanged. No uncertain sends/jobs were replayed.

Live isolated acceptance: configured Moeka model -> media_info pixels -> typed
stagecraft_build -> Codex --image -> independent four-worker stdio verification
(twelve roundtrips) -> offline Docker capture -> Wonhee outbox -> local captured
send acknowledgement. Output is a 3-second 1280x720 H.264 video, 72 frames at 24 fps,
zero browser errors. Repeated delivery polling sent nothing twice; stagecraft_show
reflects the current sent acknowledgement. No Telegram API test message was sent.
Evidence: /mnt/arteta/stagecraft/deployments/direct-acceptance-v2 and sibling
stagecraft/docs/DIRECT_MOEKA.md. Phone playback remains creator acceptance.

Final verification: 908 Wonhee tests pass (4 live/GPU tests deselected),
105 focused Moeka tests pass, and 37 Stagecraft tests pass.

## Reliable revisions (owner-approved 2026-10-25)

`stagecraft_build` accepts paired explicit width/height (overriding orientation),
dashboard_aspect_ratio, caption_above_dashboard, forbidden_text and base_job_id.
Unspecified revision settings and references inherit from the project-owned parent.
`stagecraft_list` lists only the active project's recent jobs and parent relationships.
The queue freezes resolved requirements and parent source hashes. The coding brief
includes the latest bounded schema-valid parent bundle, production settings,
references and diagnostics; failed parents are editing input only. Each revision
gets two coding attempts. Exhausted retries fail immediately, except capture reuse
for delivery recovery. Existing jobs/artifacts are not rewritten.

Video references use Wonhee's cached CPU contact_sheet node, preserve original asset
provenance, and pass bounded hash-checked image files to authorship. Queued responses
report requested settings, conversions, parent and limitations. HTTP benchmarking is
unsupported; retain the request in the project brief without numeric speed claims.

Requested browser layout uses data-demo-id="dashboard" and "caption". At every
captured frame the trusted renderer measures DOM geometry, visibility, viewport and
clipping ancestor containment with one CSS pixel tolerance; checks aspect ratio,
caption above dashboard and forbidden visible DOM text. State metadata is not layout
evidence. Existing exact fixture assertions remain exact and report all failures
together. Independent encoded dimensions/duration checks still gate delivery.
Reports retain measurements and failure screenshots. Failures retain stage, code,
explanation and diagnostics path. Offline execution and idempotent outbox remain.

Reliable revisions status: built, independently aligned 2026-10-25. All review
findings resolved. Verification: 911 Wonhee tests (4 live/GPU deselected), 41
Stagecraft tests and Stagecraft Ruff checks pass. Real cached CPU video conversion
is covered. Isolated latest-failed revision exercised a blocked rendered-layout
attempt followed by one successful repair; all 480 frames passed at 1000x1600,
20 seconds, embedded 960x600 dashboard and caption above it. One local delivery
was acknowledged without Telegram sends or duplicate staging. Full evidence is in
sibling Stagecraft docs/VALIDATION.md §Reliable revision acceptance.

## Desktop asset and paragraph profiler (owner-approved 2026-10-26)

New Stagecraft builds default to a landscape desktop asset only (1280x800); the
large decorative graph and dataset-driven profiler share the viewport. No external
caption, social chrome or portrait presentation is authored by Stagecraft by default.
Legacy explicit caption/layout settings remain readable for existing jobs.
`stagecraft_build(caption=...)` supplies Wonhee's finishing caption; omitted means
the title, empty string explicitly requests the raw desktop. Caption text lives in
the queue envelope, never the generated ProductionSpec. Wonhee completion uses the
existing CPU compose_video engine with a new desktop_caption template: a simple
caption above a contained landscape clip on 1000x1600. No speech, LLM or GPU is
needed. Final delivery contains the composed video, with source-capture and finishing
provenance retained. Idempotent receipts compare original capture hash, caption and
trusted route before any re-render; caption recovery can reuse the cached compose.
Legacy completion without caption still delivers original bytes.

A reproducible standalone CPU-only benchmark script generates seeded, unique random
paragraph-sized strings and sends identical JSON echoes through persistent HTTP/1.1
(loopback, TCP_NODELAY, no TLS) and NDJSON child pipes. Alternate transport order,
record every paired message's payload identity, trial and measured RTT; verify every
returned string. Export paragraphs, per-message CSV, summary, hashes, and bounded
plot/replay JSON. No selected/fabricated latency or generalized agent speed claim.
The profiler replays measured samples; decorative mesh activity is staged and
explicitly distinguished in provenance. This is a fixed offline benchmark, not a
new arbitrary backend execution capability. Wonhee owns final caption composition.
`profiler_asset_id` selects project-owned graph-ready JSON (<=120 KB) with summary,
pairs, histogram and scope. Its content hash and asset identity are frozen in the
request. It is supplied to authorship in initial.benchmark and remains authoritative
through generation/repair: generated initial state or actions cannot replace the
dataset. Unspecified revision datasets inherit. This allows measured profiler
numbers from supplied evidence without granting benchmark execution to the builder.

Desktop/profiler extension status: built, independently aligned 2026-10-26. Verified
45 Stagecraft tests plus Ruff and 931 Wonhee tests (4 live/GPU deselected). Live
acceptance used 512 random paragraphs, five trials, 5,120 checked observations; strict
schema validation and immutable coding inputs passed. A 480-frame 1600x1000 desktop
capture shared graph/profiler space. Actual Wonhee CLI finishing made the separate
1000x1600/20-second captioned video; repeated completion produced one final delivery,
acknowledged only by a local sink. Failure recovery never staged raw footage.

## Conversational production without manual scripts (owner-approved 2026-10-26)

Moeka must be able to complete the paragraph dataset → desktop graph/profiler →
Wonhee caption → outbox flow using tools alone. No user-authored code, JSON imports,
filesystem debugging or manual source patches are required.

stagecraft_dataset(title?, paragraph_count=512, trials=5, seed=24026) queues the
fixed offline paragraph benchmark. Project/request/route are host-bound. Its worker
checkpoints a hashed dataset, imports corpus/samples/plot JSON/manifest/ZIP as
project-owned references, records measured scope in project memory and stages the
ZIP once. stagecraft_show returns measured summary and asset IDs. No coding agent
is used for dataset-only jobs.

stagecraft_build gains benchmark=auto|refresh|off (default auto). Browser stdio
builds automatically create paragraph data when no supplied/inherited dataset is
available; refresh requests a new run, off disables creation. A supplied dataset
and refresh cannot be combined. Generated data is validated, registered and frozen
in a resolved spec before authorship. Retries reuse complete datasets. Original
submission identity and parent artifacts remain immutable. Legacy portrait/caption
revisions default back to desktop geometry unless explicitly requested otherwise.

The shared desktop component provides the graph and data-driven profiler, including
finite measured stats, a sampled trace and complete histogram. It accepts ordinary
visual direction/custom revisions; the builder may reuse it rather than rebuilding
charts/layout. Rendered checks require visible, non-overlapping mesh/profiler panels
with substantial dashboard area when a measured profiler is requested. Authorship
and offline execution remain separate, and coding repair remains bounded.

Status includes actionable recovery guidance and bounded diagnostics. Wonhee imports
successful preview images and bounded failure screenshots as project-owned assets;
Moeka can inspect them via media_info and revise with base_job_id. It must diagnose
and act on recoverable failures itself rather than ask the creator to read logs or
edit code. Exhausted coding requires a new revision; delivery recovery reuses capture.
Every generated dataset and final delivery preserves content hashes and source scope.
Only the finished captioned video (or explicitly requested raw desktop) is delivered.
Trusted route, project ownership and the idempotent outbox boundary remain unchanged.

Conversation-only extension status: built, independently aligned 2026-10-26. Actual
configured Moeka model + authenticated MCP queued one video from a normal request;
worker-generated data, Codex-authored stdio, shared desktop capture, Wonhee finishing
and a deduplicated local delivery all passed without manual source edits. A status
request found the job and inspected its real preview/contact sheet through tools.
Dataset-only MCP→worker→ZIP delivery and same-request corrective revision identities
are regression-tested. Final verification: 50 Stagecraft tests, 935 Wonhee tests
(4 live/GPU deselected), 105 focused Moeka tests and changed-surface Ruff checks pass.
Moeka's changed MCP module has no new type-check errors; 3 pre-existing environment/
OAuth typing errors remain identical to the unchanged HEAD baseline. Detailed
acceptance evidence is in sibling Stagecraft docs/VALIDATION.md.

## Caption corrections from production feedback (2026-10-26)

Status: built (aligned 2026-10-26), within the existing CPU finishing area.
Evidence: project 2 feedback memory 20 and conversation messages at 05:47–05:49 UTC
request persistent short captions with varied fonts/sizes. Submissions 1–3 passed
empty params while storing copy only in the brief; submissions 2 and 3 returned the
same output. A finished portrait asset was reused as a desktop input.

The brief is context, never executable render options. `make_submit` exposes typed
`caption` or `caption_lines` (one to four strings: hook, metric, proof, caveat),
resolves these into explicit desktop_caption slots and returns the effective copy.
Existing params.title / params.slots remain supported. Empty caption submissions,
unknown slots, conflicting copy and ignored editing options fail before queueing.
Discovery explains the actual slots and gives an example. Missing copy produces an
actionable correction for Moeka, rather than a successful blank video.

Desktop finishing contains the complete original landscape video. A caption revision
resolves a prior Wonhee finished asset back to its project-owned original desktop
using recorded hashes or submission lineage; it never recursively embeds a portrait
finish. Missing, cross-project or cyclic lineage fails before queueing. The resolved
source and effective caption appear in status. Stagecraft status points caption-only
changes to Wonhee finishing, without consuming a coding attempt.

All caption layers are static for the entire source duration. The hierarchy uses
bundled headline, numeral, sans and mono fonts above the desktop window. Text may
shrink only to 32px on the 1000x1600 canvas; copy that cannot fit fails with a shorten-
copy error before queueing, rather than silently producing tiny or clipped text.
Existing single-caption finishing uses the larger shared caption area. No GPU,
speech model, new dependency or automatic production replay is needed.

Validation: configured Moeka model + real MCP accepted an ordinary caption revision,
provided all four typed lines and selected the original desktop in an isolated store.
CPU rendering produced 1000x1600/20s; all 480 caption regions were checked and start,
middle and end frames visually inspected. One local delivery, no Telegram send.
Independent final alignment review is clean. Detailed evidence is recorded in sibling
Stagecraft docs/VALIDATION.md §Moeka caption-feedback acceptance.

## Readable access to measured benchmarks (2026-10-26)

Status: built (aligned 2026-10-26), within the existing paragraph dataset and profiler area.
The creator requests measured paragraph RTTs behind theatrical mesh language, rather
than the fictional 847× in the pasted draft. Existing 512-paragraph/5-trial evidence
must be discoverable without guessing an asset ID or asking the creator to run code.
`stagecraft_benchmarks` is a project-bound read-only CLI/MCP action. It lists bounded
validated profiler datasets, their summary and sample pairs, workload/method, scope,
ratio phrasing and project-owned evidence asset IDs. Hash-linked manifest/source/raw
CSV evidence is distinguished from a plot JSON alone. Missing/corrupt evidence is
reported explicitly, never silently promoted to a verified measurement. No build,
benchmark or delivery is started. Fresh measurement remains stagecraft_dataset.
Moeka checks this tool before proposing simulation or remeasuring, reuses the selected
profiler_asset_id, and may use theatrical interface language while presenting actual
RTTs/ratios as local echo results. A transport microbenchmark does not measure a
wrapper that was not run, thousands of agents, zero-copy, or live mesh latency.
The existing legacy measurement is registered with its full hash-checked source and
raw data; historical assets remain immutable. No video build is authorized here.

Acceptance: project 2 now owns full evidence assets 26–32, with profiler asset 31
(the exact same plot bytes as legacy asset 15). Hashes of all original artifacts
were checked before registration; deliver=False, no benchmark/build/send. The
configured Moeka model called the real MCP discovery tool in an isolated snapshot,
reported the actual medians, ratio and sample pairs, and found complete evidence.
Independent alignment review is clean. Evidence lives in
/mnt/arteta/stagecraft/deployments/benchmark-discovery-acceptance/ and
/mnt/arteta/stagecraft/deployments/measured-benchmark-access/.

## Two presentations of one mesh story (2026-10-26)

Status: built (aligned 2026-10-26), requested by the creator. Build shared footage and two reviewable
variants; leave production assembly/delivery to Moeka when the creator requests it.
Stagecraft exposes presentation=mesh_story, a deterministic 20-second built-in
browser dashboard using saved measured profiler data and distinctly labelled
scripted logical-agent/throughput telemetry. Its tall desktop (1080x1440 by default)
contains substantial graph/profiler areas, phase changes, terminal reveal and
interactive cluster selection/pause controls. It needs no new coding dispatch,
process fleet or GPU. Explicit dimensions still win. The standard presentation is
unchanged. Supplied benchmark bytes/ratios are never changed to match marketing copy.

Wonhee adds desktop_reel (1080x1920) CPU finishing. make_submit accepts actual
caption, caption_lines, or caption_cues=[{start_seconds,end_seconds,text}]. Cues must
be nonempty, ordered, non-overlapping, contiguous from zero, and cover the source
exactly. They cannot mix with other caption copy. Validate readability, timing and
slot conflicts before queueing and again in the worker. A static caption lasts the
whole source. Cues cut at exact boundaries without fades, above a fully contained
source. Timed captions use source-time spans; trimming/looks remain disabled.
Status returns effective cues/source. The existing 10:16 style remains compatible.
Source recovery supports both finishing styles and a verified raw portrait Stagecraft
capture, without treating unknown finished portraits as original footage.

One immutable raw capture supplies both presentations and keeps their underlying
simulation identical. A uses one static hook; B uses five four-second reveal cues.
The profiler shows the real local echo medians/ratio (5.514x, not unmeasured 5.6x),
while logical agents and traffic rates are simulated. No publishing or Telegram
send is authorized by this implementation request. Save project-owned assets and
source-linked assembly recipes so Moeka needs no filesystem/JSON/debugging from
the creator to finish later. Local isolated renders prove both variants build.

Multiple cuts from one authenticated turn use make_submit variant (a stable 1–40
character alphanumeric/underscore/hyphen label) to derive distinct request keys.
The host still binds the original request and route. Same variant+same inputs reuse
the job; changed input under the same variant is rejected. Legacy unlabelled keys
remain compatible, and old stored options normalize newly added defaults on retry.

Shared-source build uses stagecraft_build prepare_only=True to import its verified
raw desktop without composing a caption or staging any delivery. Omitted caption
is suppressed in this mode; explicit nonempty caption is rejected. The durable job
returns prepared asset IDs and phase=prepared. Repeating completion reuses the
source-key import; no outbox receipt or send is created. The default remains normal
delivery. This lets Moeka build once and later make two final cuts without sending
an intermediate raw video. Dataset-only jobs do not accept this video preparation flag.
The built-in mesh story is 20 seconds and 3:4; different supported 3:4 dimensions still
win over defaults. A different story duration/aspect uses a custom standard scene.

Validation: independent final alignment review is clean; full Wonhee suite 979 passes
and 35 focused current preparation/caption checks pass. Stagecraft 54 tests/Ruff pass.
Final raw source 41, reviewed previews 42/43 and post-copy 44 are saved in project 2
with exact assembly recipes, without production delivery. Configured Moeka model
retrieved these recipes and queued both variants in one turn in an isolated store;
worker output hashes match the reviewed previews. See sibling Stagecraft
VALIDATION.md §One mesh, two presentations for capture/probe/frame evidence.

## Inspecting a blank historical capture (2026-10-26)

Status: built (aligned 2026-10-26); production evidence from job 7bd8ca786af9297f845e5dcd99de08b4.
That generated scene mounts MeshProfiler without state.benchmark. The shared
component returned a placeholder, leaving working counter/inspector overlays over
an empty viewport. Historical capture required profiler panels only when benchmark
data existed, so the placeholder passed mechanical checks. The job predates the
prepared A/B sources. Preserve its original artifact and record the diagnosis.

MeshProfiler must throw a useful dependency error when mounted without data/graph;
capture checks browser errors immediately after reset and every frame, so bounded
repair receives the cause before encoding/delivery. Coding guidance offers this
component only with supplied measured data, and calls for an independent visible
graph/profiler for simulated-only scenes. No silent fallback to invented measurements.

Moeka also hit its filesystem boundary reading external project sources and aborted
after five errors. New stagecraft_review is a project-bound read-only MCP/CLI action:
return bounded source excerpts, file hashes, initial-state keys, benchmark/graph
presence and sampled actual layout measurements alongside existing diagnostics.
Never expose arbitrary paths/reads or weaken the workspace boundary. Historical
mechanical success is not visual approval. Moeka uses this tool for source diagnosis
rather than repeated denied read_file calls. No production replay or send occurs.

Source-review responses place dependency/capture facts before compact excerpts and
accept source_file from a fixed basename allowlist for detailed excerpts. This keeps
critical facts within Moeka's 16000-character result budget. A regression guards
summary ordering and size. The configured Moeka model successfully diagnosed the
missing dataset and independent counter through the tool, with no host file reads.
Validation: 981 Wonhee tests plus 3 focused review/schema checks; 56 Stagecraft tests and
Ruff pass. Real Docker missing-data capture fails; 3s/72-frame graph/profiler validation
at story time 10–13s passes. Snapshot 20261026T174818Z; all affected services active.
No production replay or delivery was created. Detailed evidence and root-cause
report are in /mnt/arteta/stagecraft/deployments/moeka-render-investigation/.
