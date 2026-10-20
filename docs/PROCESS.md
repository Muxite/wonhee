# How we change wonhee: design first

Status: approved 2026-10-20.

Without a written design, nobody can say whether code is right or wrong. So the design comes first, the code is
built to it, and when the two disagree a fixed rule decides which one changes. This page is that rule, plus the
shortest lifecycle that keeps it true.

## 1. What outranks what

```
VISION.md  >  DESIGN.md + the area specs it links  >  tests  >  code  >  README / how-to docs
```

| Layer | Holds | Changes when |
|---|---|---|
| [VISION.md](VISION.md) | Goals, stages, "good", never-list | The owner says so |
| [DESIGN.md](DESIGN.md) + area specs | Contracts (ports, params, defaults, invariants, outputs), the reasons behind them, status | Before the code it governs |
| Tests | The design, made executable | With the design, before or with the code |
| Code | One implementation of the design | After the design |
| README, `docs/production/` how-tos | How to use it, and measured numbers | After the code ships |

A design states intent. It doesn't hold volatile facts: counts, test totals and today's model of choice go in
commits, `docs/production/` or the drift register.

Every design section carries a **status**:
- `proposed`: written, not agreed.
- `approved`: agreed, and the code must follow.
- `built`: code matches, with a commit or test as evidence.
- `superseded`: replaced, with a link to the replacement.

## 2. Code wrong or design wrong?

When code and design disagree, ask these in order and stop at the first yes:

1. **Is the section only `proposed`?** Then nothing is wrong yet. Either approve it or leave it.
2. **Does the design contradict VISION.md?** Then the **design is wrong**. Fix it, citing the vision line.
3. **Is there newer evidence against the design?** That means a measurement, a live run, a test of real media, or an
   owner ruling, dated after the section was written. Then the **design is wrong**. Update it, citing the evidence,
   then align the code.
4. **Is the design ambiguous or silent, so that both readings are defensible?** Then the **design is wrong**
   (underspecified). Write down the reading that serves the vision, then check the code against it.
5. **Otherwise the code is wrong.** Fix the code and add or adjust a test that names the section.

Code that does something no design mentions is **unowned**, not wrong. Either write it into the design (it serves a
stage) or delete it. Unowned code isn't left as it is.

Being wrong is a spectrum. A default that is off by a little is a detail, while a broken invariant is a defect.
Classify honestly and fix the expensive ones first.

### Who decides

| Change | Who |
|---|---|
| Code fixed to match an approved design | Anyone (an agent included). Just do it |
| A design detail changed on evidence (a default, a threshold, a step order) | Anyone. Cite the evidence in the section and mention it in the report |
| A new contract, a new stage or area, an item added to or removed from the never-list, a vision change, a new paid or public dependency | **The owner approves** before code is written |
| A design section and its code deleted | The owner, unless the section is `proposed` and unused |

## 3. The lifecycle, sized to the change

Pick the smallest lane that fits. When in doubt, go one size up.

**S: fix to match the design** (a bug, a crash, drift where the design already says what's right)
1. Name the section the code breaks.
2. Write a failing test, then fix the code.
3. `uv run pytest`, then commit `fix …`.

**M: new behaviour inside an approved area** (a new param, knob, check, node in an existing family, or a changed
default)
1. Edit the area spec first: the contract, why, and status `approved` if it's within the vision and the "who decides"
   table allows it.
2. Write tests from the spec text, not from the code you're about to write. A separate agent writing them is better
   still (held-out tests).
3. Build to the tests.
4. Verify. Pixel and model work also gets a live check on real media, with numbers recorded.
5. Set the section to `built`. The design and code may share one commit, but the design diff must make sense
   without the code.

**L: a new area or stage, a new external service, anything public, paid or irreversible**
1. Write a spec in `docs/specs/YYYY-MM-DD-<topic>-design.md` (status `proposed`) and link it from DESIGN.md.
2. The owner approves it.
3. Write a plan (`docs/plans/`), which is disposable.
4. Build in waves along the M lane.
5. Run the S lane on whatever the live run finds.

**Done means:** the tests pass, the live check is recorded where one applies, the section says `built`, the drift
register has no new open row for it, and the README describes how to use it.

## 4. Keeping it honest, quickly

- **Drift register** ([DESIGN.md › Drift](DESIGN.md#drift)): every known mismatch, its verdict (code wrong, design
  wrong, unowned) and what was done. When you find drift you can't fix now, add a row rather than leave it silent.
- **Audit at milestones:** after a large wave, an agent compares each `built` section with the code and adds what it
  finds to the register. Run it read-only and fan it out per area, which makes it cheap.
- **Tests cite the design:** a test that guards a design rule names its section in the docstring
  (`design: photo §ladder`), so a failing test tells you which promise broke.
- **Live findings feed the design:** when a live run disproves a rule (for example the low-key photo lifted +0.5
  stop), the design changes first and the code follows (rule 3 above).
