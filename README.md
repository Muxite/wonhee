<div align="center">

# wonhee

**An AI production studio for one creator.**

From “what should I post?” to something worth sharing, and then to what the numbers say about the next one.

[The vision](docs/VISION.md) · [What’s built](docs/DESIGN.md) · [Setup & usage](docs/USAGE.md) · [The art library](docs/ART.md) · [Tools index](docs/TOOLS.md) · [For agents](AGENTS.md)

</div>

![wonhee — a creative studio for one. Ideas, footage and creative direction come together in one production workflow.](docs/assets/wonhee-studio.svg)

Making a post involves a lot more than pressing record. There is the idea, the research, the footage you
forgot you shot, the edit, the captions, the sound, and the next round of changes. For a solo creator,
every handoff lands back on the same person.

**wonhee brings that work into one creative pipeline.** It combines research, planning and media tools
with AI assistants that can carry out the work. The goal is to give one person the support of a small
production team, while keeping their taste and creative direction at the centre.

## It works: a few numbers

wonhee runs a real Instagram account (a student creator). Every post gets a written
post-game analysis, so the studio learns from results and not only from taste.

| | |
|---|---|
| Best reel | **1.5M+ views** and 800K+ viewers in 3 days, from a single static shot of 6.7 s with no effects |
| Where it came from | 80% from the Reels tab, 11% Explore, under 3% from followers: strangers, not the existing audience |
| Held attention | **60% watched to the end**, skip rate 31.7%, and shares at about 2% of views (32K) |
| Typical reel | a few thousand views |
| Library | 90+ reels and posts stored and analysed (speech, shots, on-screen text, look), 200+ deterministic art effects |

What those posts taught the pipeline so far:

- **One shot beats a busy edit.** The simplest cut won by two orders of magnitude over the same account's 10 s and 20 s
  edits. wonhee now defaults to the minimal cut and makes effects opt-in.
- **The premise has to land by about 2.5 s.** Likes by video second peak there, and half of a weaker post's viewers were gone
  by 4 s.
- **Reach and followers are different problems.** Follows stayed near 0.01% of viewers at 110K, 230K and 810K viewers alike, so
  growth needs a deliberate follow-up, not just a bigger hit.
- **Consistency is a pipeline problem.** The next step is a bank of finished posts and a weekly batch session, so nothing
  depends on a daily burst of effort.

The numbers come from Instagram's own insights, pasted in by the creator. Per-post analyses are kept private.

## What wonhee brings to the studio

| The creative challenge | What wonhee can do today |
|---|---|
| **“I have no idea what to make.”** | Study relevant posts, surface patterns, and turn a brief or reference into ideas and shot lists. Ideas are saved with their references so they can be picked up later. |
| **“The clip I need is somewhere in my footage.”** | Import public Google Drive files and folders, index footage, and search shots by description. |
| **“I shot it. Now comes the editing.”** | Build styled reels with timed captions, animated text, photo insets and optional pause trimming. Cut stills into a montage with transitions, effects and synthesized sound. |
| **“I don’t have a green screen.”** | Key out a plain wall (or a real green screen) by detecting the backdrop colour, and drop yourself into any background, such as a drawn plane window or cabin. A person mask protects a white shirt on a white wall. |
| **“These photos need to feel like they belong together.”** | Apply or develop photo looks, adjust colour, and compose posters, carousels and collages. Help placing text on a photo and judging whether the photo can carry it. |
| **“I need a backdrop, a card or a cover.”** | The [art library](docs/ART.md): about 200 deterministic image effects (physics, light, dot-matrix, graphs, colour grades, edits of real photos) with palettes, seeds and no generative AI. |
| **“Did it work, and what do I do next?”** | Store any public reel by URL with its metrics and analysis, take a reference apart into hook, shots, text and sound, and keep a post-game analysis for everything published ([tools index](docs/TOOLS.md)). |
| **“The sound and subtitles need work.”** | Transcribe speech, translate subtitles, reduce noise and balance loudness. |
| **“My work is code or data. How do I show it?”** | Turn repository structure, project history and other data into visuals and credited card sets, or record small technical demos (terminal skits, graphs, browser apps) in shared offline containers. |

Projects also keep ideas, preferences and feedback together, so a creative assistant can pick up an
ongoing project with its context intact.

## The bigger picture

![The intended wonhee workflow: discover what resonates, plan an original idea, bring in real footage, make the edit, check the result, and deliver it to the creator.](docs/assets/wonhee-workflow.svg)

The ambition is one connected journey: **discover → plan → capture → make → check → deliver → learn.**
A reference should help shape an original idea. A plan should guide the shoot. The footage should flow
into an edit, and feedback should inform the next version.

That is the direction of the project. Today, many of the research, planning, editing, art and import tools
are built. The next priority is to connect reel editing, comprehensive output checks and phone delivery
into a complete workflow. The Telegram integration is deployed, with an original video successfully
delivered and played back on a phone; the full edited-reel round trip still needs live validation.

## Built around the creator

**Your footage and your point of view.** Photo and video edits work with real captured media. References
help explore a direction while the result stays original. Nothing is generated by an image model.

**Conversation backed by tools.** In the current Telegram setup, moeka is the conversational front door
and wonny is the dedicated creative assistant. Wonhee provides the media workspace and production tools
behind them, over a small versioned contract so a change on either side is reported instead of failing
quietly ([bridge contract](docs/specs/2026-11-03-bridge-contract.md)). Other AI assistants can connect
through MCP, a standard way for assistants to use tools. For technical demos, ask Moeka for a Stagecraft
build: it records a small app, graph or terminal skit and returns the MP4
([direct workflow](docs/specs/2026-10-25-direct-moeka-design.md)).

**Local work where it makes sense.** Media processing runs on the machine, including the art library and
the chroma key. Research and some AI tasks can use external services, with spending limits on paid
production runs.

**You decide what goes out.** Wonhee prepares files for you to review and share. Social posting stays
with the creator.

## Where it’s headed

- **A complete reel workflow:** an integrated editing timeline, full file checks, and phone inbox and outbox.
- **A more personal creative voice:** style memory for the rhythm of captions and cuts.
- **More help before the shoot:** a director that turns an idea inbox into plans, plus small prop projects for skits.
- **Art for video:** the art library’s looks and effects, applied to motion as well as stills.

Wonhee is an actively developed personal production system. Its aim is practical: give a solo creator
more room to make things by taking on the work that surrounds making them.
