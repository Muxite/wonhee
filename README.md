# wonhee

The flagship for influencer-side automation: find what is trending in a niche
(now, or 6 months to 2 years back for throwbacks), turn it into a shot list you
can just film, edit raw recordings the way most of the platform currently does,
and build small disposable prop projects for skits.

wonhee is agent-centric (moeka drives it over Telegram) but the work runs as a
fixed pipeline over a relational store, so every output traces back to the run,
step, and source content that produced it.

Sub-projects, in build order:

1. **Trend scout**: [design](docs/specs/2026-10-15-trend-scout-design.md)
2. Editor
3. Prop builder
4. Director / idea inbox

The job-application form filler that used to live here is now `../awork-fill`.
