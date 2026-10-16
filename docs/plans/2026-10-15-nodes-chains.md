# wonhee nodes, chains and translation: implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development, with **Sonnet** implementer and reviewer agents as on scout-v1. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Every tool becomes a typed node. An agent or a person writes a chain once (JSON). wonhee runs it, piping each node's output into the next with nobody watching, and can return immediately with a run id. Translation is the first node set.

**Spec:** `docs/specs/2026-10-15-toolkit-design.md`, sections R0 (nodes/chains), R7 (translation nodes) and the reworked R2. Approved 2026-10-15. The user called it "a start, likely expanded later depending on how it feels in practice". So: keep the core small and the node interface easy to add to, and gold-plate nothing.

**Repo/branch:** `/home/muk/projects/wonhee`. Create `toolkit-v1` off `scout-v1` (HEAD e8c2441). Local only, **never push**.

## Context
- Reel 3 (DcSv-shxIDR, a 108 s Mandarin skit) was translated with a throwaway script (`scratchpad/translate_proto.py`): 63/63 timed lines for $0.003.
- That run exposed two gaps:
  - On-screen text was read once per shot. A one-shot video gave one reading.
  - The vision model misread 阿妩 as 阿姨.
- The user wants translation as a node that agents can string with other nodes, deciding the steps up front without supervising each one.
- Out of scope for this plan, each getting its own plan later: the other toolkit parts (edit ops R1, saved `basic` chain R2, checks R3, phone page R4, media_info R5, style R6). This plan builds the node/chain core plus `burn_subtitles`, which is the only R1 piece translation needs.

## Global Constraints
- Writes only via `wonhee/db/repo/*`. Reads via `wonhee/db/queries.py` or inline SELECTs in render code.
- Model calls only via `wonhee/llm.py` (`ctx.llm(...)`). Untrusted text (captions, transcripts, subtitles, OCR) goes into prompts only via `llm.data_block()`.
- Every subprocess (ffmpeg, yt-dlp) gets `procenv.clean_env()`. The single exception is the detached `wonhee chain exec` child (it is wonhee itself), described in Task 8.
- Keys are project-scoped (memory rule). Library code never reads `.env`; only entry points call `envfile.load_entry_env`.
- Tests never hit the network or a real model. Use the `fake_llm` fixture (patches `llm._backend`) and the `tests/fakes.py` deps. GPU tests are marked `gpu`, live tests `live`.
- Commits: lowercase, 3–8 words, no AI co-author trailer. Never `git add -A`; add files by name.
- Never print secrets. Never commit `.env`, `data/` or scraped corpora.

## Review Focus (failure modes no single task's happy path covers)
1. **Resume after a failed node reads the current upstream outputs.** `resume RUN --from t` must feed `t` the outputs of the *current* `s` step, never a superseded one. Test in Task 3: `test_resume_reads_current_upstream_outputs`.
2. **The cache never serves a weak or failed result, and never crosses node versions.** Test in Task 3: `test_cache_skips_weak_and_version_bump`.
3. **A bad chain spends nothing and records nothing.** Every validation error is listed at once, and no intent/project/run rows are created. Test in Task 2 (pure) and Task 8 (`test_chain_run_invalid_records_nothing`).
4. **No-audio / no-speech / no-text videos flow through the chain.** `speech_subs` gives empty lines, `translate` of empty lines makes no model call, `to_srt` writes an empty file, and `burn_subtitles` copies the video unchanged. Test in Task 4 and Task 6.
5. **The detached child sees the same home and keys as the parent and never runs twice.** `chain exec` on a run that is already `done` is a no-op. Test in Task 8.

---

## Shared definitions (the implementers of Tasks 2–8 all receive this block)

```python
# wonhee/nodes/types.py
PORT_TYPES = ("content", "video", "image", "subtitles", "text", "file")
# Ref is a str: "content:<id>" for content; a 64-hex blob sha for the rest.
SHA = re.compile(r"^[0-9a-f]{64}$"); CONTENT = re.compile(r"^content:(\d+)$")
def parse_type(t: str) -> tuple[str, bool]: ...   # "subtitles?" -> ("subtitles", True); unknown -> ValueError
def ref_ok(type_: str, ref: str) -> bool: ...     # content -> CONTENT match; others -> SHA match

# wonhee/nodes/base.py
@dataclass(frozen=True)
class Node:
    name: str
    version: int
    inputs: dict[str, str]      # port -> type ("type?" = optional)
    outputs: dict[str, str]
    params: type[BaseModel]     # extra="forbid"
    fn: Callable[[Ctx, dict[str, str], BaseModel], dict[str, str]]   # returns port -> ref
    doc: str                    # one line, shown by `node list` / nodes_list
# A node marks its result weak with ctx.flag(...) plus setting ctx.weak = True (new Ctx field, default False).

# wonhee/nodes/registry.py
NODES: dict[str, Node]   # filled by importing wonhee.nodes.subs, .translate, .screen, .grab, .burn
def get(name) -> Node    # KeyError("unknown node 'x'; known: a, b, …")

# wonhee/nodes/subtitles.py
class Line(BaseModel): id: int; start_ms: int; end_ms: int; text: str; source: Literal["speech", "screen"]
class Subtitles(BaseModel): language: str | None = None; lines: list[Line] = []
def load_subs(media_dir, sha) -> Subtitles
def save_subs(conn, media_dir, subs) -> str        # repo.save_bytes(..., "application/json"); returns sha
def save_text(conn, media_dir, text) -> str         # "text/plain; charset=utf-8"
```

**Chain JSON** (spec R0): a list (or `{"nodes": [...], "cost_cap_usd": 1.0, "no_cache": false}`) of
`{"id", "node", "in": {port: "<id>.<port>" | literal}, "params": {...}, "cache": true}`.
- Ids match `^[a-z][a-z0-9_]{0,31}$`.
- Literals are a sha or `content:<id>`, or a local path. Paths are resolved to refs by the service *before* validation and storage (Task 8), so stored chains hold refs only.

---

### Task 1: Branch, migration 0006, repo + queries for node outputs and cache
**Files:** `wonhee/db/migrations/0006_nodes.sql`, `wonhee/db/repo/nodes.py`, `wonhee/db/repo/__init__.py`, `wonhee/db/queries.py`, `tests/test_schema.py`, `tests/test_repo_nodes.py`.

- [ ] `git checkout -b toolkit-v1` (from scout-v1).
- [ ] Migration `0006_nodes.sql`. Start with the header `-- wonhee:foreign-keys-off` and copy the 0005 pattern.
  - Rebuild `project`, adding `'chain'` to the kind CHECK.
  - Create the node tables:
    ```sql
    CREATE TABLE step_output (
      step_id INTEGER NOT NULL REFERENCES step(id), port TEXT NOT NULL,
      type TEXT NOT NULL CHECK (type IN ('content','video','image','subtitles','text','file')),
      ref TEXT NOT NULL, PRIMARY KEY (step_id, port)) STRICT;
    CREATE TABLE node_cache (
      key_sha256 TEXT NOT NULL CHECK (length(key_sha256)=64), port TEXT NOT NULL,
      type TEXT NOT NULL CHECK (type IN ('content','video','image','subtitles','text','file')),
      ref TEXT NOT NULL, node TEXT NOT NULL, node_version INTEGER NOT NULL,
      created_by_step INTEGER NOT NULL REFERENCES step(id), PRIMARY KEY (key_sha256, port)) STRICT;
    INSERT INTO flag_code(code, meaning) VALUES
      ('CACHED', 'result reused from the node cache; no work was done'),
      ('TRANSLATION_GAP', 'some lines came back untranslated and kept their original text'),
      ('NO_TEXT', 'nothing to work on (no speech or no on-screen text); output is empty');
    ```
  - The other toolkit tables (upload/outbox/media_check/style*) belong to later plans, in migration 0007+. (Ruling: they were listed under 0006 in the spec; splitting them out costs nothing.)
- [ ] Repo functions:
  - `set_step_outputs(conn, step_id, outs: dict[str, tuple[type, ref]])`
  - `put_cache(conn, key, node, version, step_id, outs)`, using `INSERT OR REPLACE`
- [ ] Queries:
  - `step_outputs(conn, step_id) -> dict[port, Row]`
  - `cache_get(conn, key) -> dict[port, (type, ref)] | None`
- [ ] Tests:
  - the migration applies on a DB at version 5 that has scout and library projects, and they survive (copy `test_schema.py`'s 0005 rebuild test);
  - `chain` kind is accepted;
  - repo round-trip;
  - a bad type is rejected.
- [ ] Commit: `add node output and cache tables`.

### Task 2: Node core: types, base, subtitles model, chain model and validation (pure)
**Files:** `wonhee/nodes/__init__.py`, `types.py`, `base.py`, `subtitles.py`, `chain.py`, `registry.py` (with an empty `NODES` plus a `register()` helper for tests); `tests/test_chain_validate.py`.

**Interfaces:**
- `chain.ChainNode` (pydantic, alias `in` → `in_`, `extra="forbid"`)
- `chain.Chain(nodes, cost_cap_usd: float | None = None, no_cache: bool = False)`
- `chain.parse_chain(obj: list | dict) -> Chain`
- `class ChainError(ValueError)` with `.problems: list[str]`. Its message is `"chain is invalid:\n- p1\n- p2"`.
- `chain.validate(chain, nodes: dict[str, Node]) -> list[ChainNode]` returns the topological order and raises `ChainError` listing **all** problems:
  - duplicate id; bad id format;
  - unknown node;
  - unknown input port; required input missing;
  - wire not `<id>.<port>`, or pointing at an unknown id or an unknown output port;
  - type mismatch: the wire's output type ≠ the input type (an optional input accepts its base type);
  - literal that isn't a valid ref for the port type;
  - a node wired to itself; a cycle (names the ids involved);
  - params failing `node.params.model_validate` (pydantic message, prefixed with the chain id).
- Order: Kahn's algorithm, with ties broken by position in the file.
- `chain.cache_key(node, inputs: dict[str, str], params: BaseModel) -> str` = sha256 of `json.dumps({"node", "version", "inputs", "params": params.model_dump(mode="json")}, sort_keys=True)`.

**Tests** use toy nodes registered in the test, one per problem class, plus:
- a chain with 3 problems reports all 3;
- order is topological with file-order ties;
- a bare list parses;
- `cache_key` is stable under dict-order changes and changes with version or params.

- [ ] Commit: `add node types and chain validation`.

### Task 3: Chain execution on the existing orchestrator
**Files:** `wonhee/pipeline/context.py` (add `weak: bool = False` to `Ctx`; `Stage` gains `params: dict = field(default_factory=dict)`), `wonhee/pipeline/orchestrator.py`, `wonhee/nodes/runner.py`, `wonhee/pipeline/registry.py`, `wonhee/service.py` (`_stages_of_run` → `registry.stages_for_run`), `tests/test_chain_run.py`.

- [ ] `orchestrator.load_request`: dispatch on `intent.schema_name`. `"scout"` → `ScoutRequest` as now; `"chain"` → `nodes.chain.Chain`.
- [ ] `run_stages`: record `start_step(..., {**stage.params, **overrides}, ...)` (scout stages have empty params, so nothing changes for them). After `stage.fn`, if `ctx.weak` and the status is `"ok"`, use `"weak"`.
- [ ] `runner.stages_for_chain(chain) -> list[Stage]`: one Stage per validated node, in order.
  - `name` = chain id, `needs` = upstream ids, `params` = the node's params.
  - `fn` = closure `_run_node(ctx, cn, node)`:
    1. Resolve inputs. For a wire, take `queries.current_step(conn, run, up_id)` → `step_outputs[port].ref`. A missing output raises `StageFailed("EXCEPTION", "...")`. A literal is used as is. Absent optional inputs are omitted.
    2. `params = node.params.model_validate(cn.params)`; `key = cache_key(...)`.
    3. If `cn.cache and not chain.no_cache` and `cache_get(key)` hits: `set_step_outputs`, `ctx.flag("CACHED", node.name)`, return `"ok"`.
    4. `outs = node.fn(ctx, inputs, params)`. Check that keys == `node.outputs` keys and each ref passes `ref_ok`; a violation raises `StageFailed("EXCEPTION", ...)`.
    5. `set_step_outputs`. Call `put_cache` only when `not ctx.weak`.
    6. Return `"weak" if ctx.weak else "ok"`.
- [ ] `registry.stages_for_run(conn, run_id)`: kind `chain` → `stages_for_chain(load_request(...)[1])`; other kinds → `stages_for(kind)`. `service._stages_of_run` and `_next_stage` use it.
- [ ] Tests use toy nodes registered in the test (e.g. `upper`: text → text; `count`: text → text; `boom` raises):
  - happy path with outputs recorded;
  - a failure stops downstream;
  - `test_resume_reads_current_upstream_outputs`: change upstream params via resume and check the downstream input ref changed;
  - a cache hit costs $0, makes no fn call (counter) and flags `CACHED`;
  - `test_cache_skips_weak_and_version_bump`;
  - the `no_cache` flag and the per-node `cache: false` both skip the cache;
  - the cost cap still stops a chain;
  - scout tests stay green (the full suite runs).
- [ ] Commit: `run chains as orchestrated steps`.

### Task 4: Subtitle nodes: `speech_subs`, `merge_subs`, `to_srt`, and the `grab` node
**Files:** `wonhee/nodes/subs.py`, `wonhee/nodes/grab.py`, `wonhee/library/grab.py` (refactor), `tests/test_nodes_subs.py`.

- **`grab` node** (`→ content, video`; params `url`, `relation="external"`, `refetch=false`; version 1).
  - Refactor `library/grab.py`: extract `grab_urls(ctx, urls, relation, refetch) -> dict[str, int]` (external_id → content_id) from `grab()` so both use it. `grab()` keeps its behaviour and tests.
  - The node cleans the URL (strip the query) and validates it with `intent.gaps.IG_URL` (else `StageFailed("NO_CANDIDATES", …)`).
  - It adds the content as `project_reference(…,'inspiration')`, or as `project_output` for `relation="ours"`, under the chain project.
  - Output: `content:<id>` plus the content's `media_sha256`. A missing media sha raises `StageFailed("DOWNLOAD_FAILED")`.
  - `cache` is not used for grab: its node is declared `cacheable=False`. Add `cacheable: bool = True` to `Node`; the runner skips the cache when it is False. (Ruling: grabbing again is how fresh metrics are fetched.)
- **`speech_subs`** (`video → subtitles`).
  - If `queries.media_analysis_row(conn, sha)` has a `transcript_sha256`, build the subtitles from that JSON (`segments` → lines, `source="speech"`). Otherwise use `media.has_audio()`; with no audio, return empty subtitles and flag `NO_TEXT`. With audio, call `ctx.deps.transcriber.transcribe(path)`.
  - Lines with blank text are dropped and ids renumbered from 0.
  - Empty output → `NO_TEXT` flag (not weak).
- **`merge_subs`** (`a: subtitles, b: subtitles → subtitles`; param `prefer="a"`, the only option for now).
  - Keep all of `a`. Drop a `b` line when it overlaps an `a` line by ≥ 50 % of the `b` line's duration **and** `difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.6`, where `norm` lowercases and strips whitespace and punctuation.
  - Sort by `start_ms`, renumber ids, `language = a.language or b.language`.
- **`to_srt`** (`subtitles → file`; `max_chars=42`).
  - Standard SRT (`HH:MM:SS,mmm`), numbered from 1.
  - Wrap at word boundaries when the text has spaces, otherwise every `max_chars // 2` characters (CJK). Empty input → empty file.
- **Tests:**
  - speech_subs reuses a stored transcript (the transcriber is not called, via a counter) and transcribes when there is none;
  - a no-audio video gives empty output + `NO_TEXT`;
  - merge drops near-duplicate overlaps and keeps distinct ones;
  - SRT timing format and wrapping for English and CJK, round-tripping the times;
  - a grab node chain with fakes (copy the setup from `tests/test_grab.py`) outputs a valid content ref and video sha;
  - `test_grab.py` still passes.
- [ ] Commit: `add grab and subtitle nodes`.

### Task 5: `translate` node
**Files:** `wonhee/nodes/translate.py`, `wonhee/scout/prompts.py` (add `SYSTEM_TRANSLATE`), `tests/test_node_translate.py`.

- **Signature:** `subs: subtitles, caption: content? → subs: subtitles, notes: text`. Params: `to="en"`, `style: Literal["natural","literal"]="natural"`, `batch=80`, `overlap=10`. Version 1.
- **Model schema:**
  ```python
  class TLine(BaseModel): id: int; text: str
  class Term(BaseModel): source: str; target: str
  class TranslateReply(BaseModel):
      lines: list[TLine]; glossary: list[Term] = []; caption: str | None = None
      gist: str | None = None; notes: list[str] = []
  ```
- **Prompt** (proto: `scratchpad/translate_proto.py`):
  - Ask for subtitle-style output in the target language, one line per id, same ids.
  - Data blocks: caption (batch 1 only), the previous `overlap` lines (source + translation, as context only, "do not translate these again"), the glossary so far, and the numbered lines `[id] start s: text`.
- **Flow:**
  - Batches of `batch` lines. The glossary is merged across batches; the first target wins on conflicts.
  - Ids missing from a reply get one retry call containing only those ids. Any still missing keep their source text, flag `TRANSLATION_GAP` with the ids, and set `ctx.weak = True`.
  - Ids the model invents are ignored.
- **Output:** the same timings and sources, `language=to`.
- **`notes` text:** `gist`, then `caption: <translated>` if there is one, then a `glossary:` list, then `notes:` bullets, joined with blank lines.
- **Empty subtitles:** no model call, empty output, `NO_TEXT`.
- **Tests (fake_llm):**
  - keeps every id and timing;
  - 170 lines → 3 batches, and batch 2's prompt contains batch 1's glossary term (inspect `fake_llm.calls()`);
  - missing ids → retry → still missing → `TRANSLATION_GAP` + weak + the source text is kept;
  - invented ids are ignored;
  - caption content is included in batch 1 only;
  - empty input → zero calls.
- [ ] Commit: `add translate node`.

### Task 6: `screen_text` node (time-sampled on-screen text)
**Files:** `wonhee/nodes/screen.py`, `wonhee/analyze/frames.py` (add `extract_every(path, every_s, dest_dir) -> list[tuple[int, Path]]` using one `ffmpeg -vf fps=1/<every_s>` pass with `clean_env()`), `wonhee/scout/prompts.py` (`SYSTEM_SCREEN_TEXT`), `tests/test_node_screen.py`.

- **Signature:** `video, context: subtitles? → subtitles`. Params: `every_s=2.0`, `max_frames=90`, `dedupe_hamming=4`, `batch=8`. Version 1.
- **Sampling:** if `duration / every_s > max_frames`, set `every_s = duration / max_frames` and note it in the step params artifact (`ctx.save_json`, role `api_response`). Frame i has time `i * every_s * 1000`.
- **Dedupe:** compute `phash_hex` per frame. A frame within `dedupe_hamming` of the last *read* frame is not sent; it extends that frame's span.
- **Vision calls:** read frames go in batches of `batch` images: `ScreenRead(frames: list[FrameText(index: int, text: str | None)])`.
  - The prompt says: give the exact on-screen text in its original language, null if none, ignore watermarks/handles. The context subtitles' text goes in a data block "speech in this video (use it to spell names and words correctly)".
- **Lines:** each read frame with text → a line from its time to the next read frame's time (or the video end). Consecutive lines with equal normalised text merge. `source="screen"`, `language = context.language`.
- **No text found:** `NO_TEXT`, empty output.
- **Tests:**
  - fixture: generate a 9 s 320x568 video of three 3 s segments from different static lavfi sources (e.g. `smptebars`, `rgbtestsrc`, `color=c=white` + `drawgrid`). In the fixture, assert their phashes differ by > 4; skip if ffmpeg is missing.
  - `every_s=1` → 9 frames sampled, exactly 3 read (dedupe), 1 batch call; three lines with the right spans;
  - a fake reply with equal text on adjacent reads merges them;
  - the context text appears in the prompt;
  - the `max_frames` cap widens `every_s`;
  - null text everywhere → `NO_TEXT`.
- [ ] Commit: `add time-sampled screen text node`.

### Task 7: `burn_subtitles` node
**Files:** `wonhee/nodes/burn.py`, `wonhee/edit/__init__.py`, `wonhee/edit/ass.py` (subtitles → ASS text; R1's captions op reuses it later), `tests/test_node_burn.py`.

- **Signature:** `video, subtitles → video`. Params: `font=None` (auto), `size_frac=0.045`, `margin_frac=0.22`, `max_chars=32`, `crf=20`. Version 1.
- **ASS style:**
  - `PlayResX/Y` = the video dims.
  - Font: `DejaVu Sans`, or `Noto Sans CJK SC` when the subtitle language is zh/ja/ko or the text has CJK characters and `font` is None.
  - White primary, black outline 3, shadow 0, alignment 2, `MarginV = height * margin_frac`, `WrapStyle 0`.
  - Text is wrapped to at most 2 lines of `max_chars`, using `\N`. `{`, `}` and `\` are escaped.
- **ffmpeg:** `ffmpeg -y -i in -vf ass=<file> -c:v libx264 -crf <crf> -pix_fmt yuv420p -c:a copy -movflags +faststart out`, with `clean_env()`. Store the result with `repo.save_file(kind="video")`.
- **Empty subtitles:** return the input sha unchanged + `NO_TEXT`.
- **Tests** (skip if ffmpeg lacks the `ass` filter):
  - burn one line onto `tiny_video`: a frame at the line's time differs from the source in the bottom third and matches (PSNR > 35 or mean abs diff < 2) in the top third;
  - duration and audio are kept;
  - ASS escaping and CJK font choice (unit-level, on the ASS text);
  - empty subtitles return the same sha.
- [ ] Commit: `add burn subtitles node`.

### Task 8: Service, CLI, MCP, detached runs, render, doctor
**Files:** `wonhee/nodes/registry.py` (import all node modules), `wonhee/service.py`, `wonhee/render.py` (`render_chain`), `wonhee/nodes/detach.py`, `wonhee/cli.py`, `wonhee/mcp_server.py`, `wonhee/doctor.py`, `tests/test_chain_service.py`, `tests/test_cli.py`, MCP tool-list test.

- **Service:**
  - `chain_run(conn, settings, deps, chain_obj, *, wait=True, progress=None) -> tuple[int, str]`:
    1. Parse, then resolve literal paths. An existing file becomes a ref: video/image → `repo.save_file(kind=…)`; subtitles → validate as `Subtitles` + `save_subs`; text/file → `save_bytes`. The port type comes from the target node's input.
    2. Validate. On `ChainError`, nothing has been written.
    3. Create the intent: `schema_name='chain'`, version 1, raw_text `"chain " + "→".join(node names)`, confirmed.
    4. Create the project: kind `chain`, title = that arrow string.
    5. `start_run` with cap `chain.cost_cap_usd or settings.cost_cap_usd`.
    6. If `wait`: `run_stages(... stages_for_chain(validated) ...)` and return `render_chain`. Otherwise call `detach.spawn(settings, run_id)` and return `"run N started in the background; check with: wonhee chain show N (log: …)"`.
  - `node_run(conn, settings, deps, node, inputs: dict, params: dict, wait=True)` → a one-node chain with id = node name.
  - `chain_exec(conn, settings, deps, run_id)`: used by the child. If the run status is `done` or `weak` → no-op. Otherwise start from the first stage without an ok/weak current step.
  - `chain_show(conn, settings, run_id)` → `render_chain`.
  - `resume` works for chains through `stages_for_run`. Add nothing to `OVERRIDE_KEYS` except `no_cache`.
- **`render_chain`:**
  - `render_run`'s step table.
  - Then **outputs** for each node no other node consumes (the chain's results). Show `id.port type`, then a preview:
    - subtitles: language, line count, and the first 3 lines `[12.3s] text`;
    - video: file path + `WxH, N.Ns, audio yes/no`;
    - file/text: the path, and for text the first 300 chars;
    - content: url + handle.
  - Paths are full paths into the media store, so an agent can open them.
- **`detach.spawn(settings, run_id) -> Path`:**
  - argv `[sys.executable, "-m", "wonhee.cli", "chain", "exec", str(run_id)]`.
  - env = `clean_env()` + every `WONHEE_*` var + the key names in `config.KEY_NAMES` (`OPENROUTER_API_KEY`, `BRAVE_API_KEY`, `APIFY_API_KEY`; add the constant if it's missing) present in `os.environ`. The parent already loaded `.env` into `os.environ`.
  - `start_new_session=True`, stdout/stderr → `settings.home/"logs"/f"run-{run_id}.log"`, `cwd=REPO_ROOT`.
- **CLI:**
  - `wonhee node list` (name, ports `in → out`, doc);
  - `wonhee node show NAME` (also prints the params JSON schema);
  - `wonhee node run NAME --in port=ref|path … [--set k=v …] [--detach]`;
  - `wonhee chain run FILE|- [--detach]`;
  - `wonhee chain show RUN`;
  - `wonhee chain exec RUN` (hidden help).
  - `ChainError` → stderr message, exit 2. Other errors follow the existing CLI rules.
- **MCP:** `nodes_list()`, `node_run(node, inputs, params={}, wait=True)`, `chain_run(chain, wait=True)`, `chain_show(run_id)`, all through `_guard`. Update the tool-list test.
- **Doctor:** a warning check listing chain runs with status `running` whose last step started > 30 min ago ("stale; resume with --set force=true").
- **Tests:**
  - `test_chain_run_invalid_records_nothing` (row counts unchanged);
  - a path literal is stored and becomes a sha;
  - `node_run` of `to_srt` on a subtitles JSON file;
  - `render_chain` shows final outputs only;
  - the detached test: write a subtitles JSON, then `chain run --detach` of `[to_srt]` via a `subprocess` CLI call with `WONHEE_HOME=tmp`. Poll the DB (≤ 20 s) until the run is `done`; the log file exists;
  - `chain exec` on a done run is a no-op;
  - CLI `node list` includes all 7 nodes.
- [ ] Commit: `add chain cli, mcp tools and detached runs`.

### Task 9: Live check on reel 3, docs, final review (controller, not a subagent)
- [ ] `uv run pytest` (all) and `uv run pytest -m gpu`.
- [ ] Write `docs/chains/translate_reel.json`: the R7 example chain with a `URL` placeholder note in README. Run it live on `https://www.instagram.com/reel/DcSv-shxIDR/`: `uv run wonhee chain run … --detach`, then `chain show`. Expected cost < $0.05; grab ≈ $0.003, and the reel's whisper transcript is reused from content 3's analysis.
- [ ] Check:
  - the English SRT has ~63 speech lines plus screen lines;
  - 阿妩 is rendered as a name (the glossary, and screen text given the speech as context);
  - the burned video plays, keeps its audio, and shows English lower-third subtitles. Pull 2 frames with ffmpeg and look at them.
- [ ] Rerun the same chain: everything except `grab` reports `CACHED`, at ≈ $0.003.
- [ ] README: a "Nodes and chains" usage section (node list, node run, chain run --detach, chain show, resume --from a node id).
- [ ] Final whole-branch review (most capable model), fixes, ledger. Merging is left to the user (no merge without approval).

## Verification
1. `uv run pytest -v`: all offline tests pass, including the new ones (`test_chain_validate`, `test_chain_run`, `test_nodes_subs`, `test_node_translate`, `test_node_screen`, `test_node_burn`, `test_chain_service`), and the existing scout/grab tests stay green.
2. `uv run wonhee node list` shows grab, speech_subs, screen_text, merge_subs, translate, to_srt, burn_subtitles.
3. The live chain on DcSv-shxIDR gives an English `.srt` and a subtitled `.mp4` from one `chain run --detach`. A second run is cached.
4. The MCP `chain_run` with `wait=false` returns a run id immediately, and `chain_show` later lists the outputs.
