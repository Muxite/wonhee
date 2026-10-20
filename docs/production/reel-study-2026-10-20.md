# Reel study 2026-10-20: what five reference reels teach

Five reels were grabbed as `content:48`–`52` (third-party, reference only; raw notes, frame crops and per-reel
`findings.json` are in gitignored `data/refs/2026-10-20/study/`). Five agents studied one reel each from frames,
the stored transcript and ffmpeg/numpy measurement; a sixth matched every text style to a free font by rendering
candidates next to the crops. This page is the learning in our words and numbers. The design it feeds is
[the reel styles spec](../specs/2026-10-20-reel-styles-design.md).

| id | Archetype | Length | Plays | Mechanism |
|---|---|---|---|---|
| 48 | Found-footage "legend" repost | 162 s, one take | 490k | superlative headline in a black top band + serif karaoke captions; payoff saved for the last 5 s |
| 49 | "Free tools you didn't know" list | 56 s, 21 shots (median 2.5 s) | 51k | visual proof first, a "yes you can" permission, numbered items, part-2 cliffhanger |
| 50 | Scrapbook explainer | 62 s, one take + ~45 insets | 902k | a picture for nearly every noun (one inset every 1.4 s), numbered chapters, one-word comment CTA |
| 51 | Weekly series ("Free Font Friday") | 30 s, 19 visual events | 118k | recurring brand, split screen for 47% of the runtime with assets swapping every 0.4–2 s, comment keyword |
| 52 | Voice-led list | 23 s, 22 cards, 0 cuts | 961k | a new colourful card every 0.9 s (0.47 s crossfades), scrolling list shows the progress |

## What they share (the rules we take)

1. **Sound is just the voice.** Every reel is speech only, with no music and no SFX. They're all at −14 to −15 LUFS
   integrated, with a loudness range of 2–3 LU (compressed speech). `audio_clean`'s −14 LUFS target is right.
2. **Cuts follow words, not beats.** Overlay changes and cuts land on speech onsets or phrase gaps (c52: 82% within
   80 ms of a voice onset against 42% by chance). Nobody cuts to a beat because there's no beat.
3. **Dead air is gone.** c49 and c50 have no pause of 150–250 ms or more anywhere: the take is silence-trimmed
   inside, not just at the ends. Talking pace is 3.8–3.9 words/s.
4. **Something changes every 1–2.7 s.** Shot, inset, card or caption: median 0.9–2.7 s between visual events, even
   on a single locked-off take.
5. **Text is timed to words.** Four styles of word-driven text: a karaoke pill moving word by word (c48), words
   appearing one at a time (c50), a kinetic one-word-per-line stack (c51), a list scrolling one line per word with
   the active line bright (c52).
6. **Layout is a headline band over the content and captions low.** Headlines sit at about y 0.11–0.23 for the
   whole reel. Captions sit at y 0.73–0.82 (c48, c50 and c51 put them lower than our 76% safe line; we stay at 76%,
   because the caption block can start at 79% on a tall phone).
7. **The hook is in the first 0–1.6 s and is text.** The title is on screen at frame 0 (c50, c52) or pops in with
   overshoot at 1.6 s (c48). Speech starts by 0.35–0.9 s.
8. **The ends are hard:** a CTA card (follow / comment a keyword / part 2), then a hard stop. None loops.
9. **Little or no grade.** Footage is mostly ungraded phone video; colour comes from the inserts and cards. c48's
   "look" is the source's: muted (chroma 8), heavy vignette (0.46), soft, magenta shadows: an old camcorder.

## Fonts (free lookalikes, rendered and compared)

| Role | Seen in | Original (guess) | Free font we'll bundle |
|---|---|---|---|
| condensed headline | 48 | Barlow Condensed | Barlow Condensed SemiBold |
| serif transcript caption | 48 | Times New Roman Bold (a tool default) | Tinos Bold |
| editorial display serif (+ italic accent) | 49, 51 | PP Editorial New | Instrument Serif Regular / Italic |
| tight grotesque (kinetic, titles) | 51, 52 | Helvetica Now / Inter | Inter Tight, Inter |
| rounded caption sans | 49, 50 | Proxima Nova / Poppins | Figtree, Poppins SemiBold |
| soft heavy serif on paper | 50 | Recoleta Black | Fraunces (SOFT 100) |
| marker / brush | 50, 51 | CapCut hand fonts | Caveat Brush, Gochi Hand |
| handwriting sub-line | 50 | CapCut handwriting | Mansalva |
| auto-caption in a box | 51 | Instagram built-in | Arimo |
| pixel badge | 50 | CLI banner | Press Start 2P |

Typography details that matter: negative tracking on display type (−0.03 to −0.05 em), tight line height (0.9),
boxes and pills behind captions, and a highlight colour for the active word.

## The pseudo-intellectual presentation

Only c48 has it, and partly by accident. The speaker produces argument-shaped nonsense in lecture cadence ("you
can't kill a person with no body"), and the editor's Times-style serif transcript captions make each line read
like a quoted proposition, so every non-sequitur lands as a failed syllogism. The serif does half the work: the
same words in a rounded sans read as a joke; in a book serif they read as a claim.

What makes the style, so we can do it on purpose (and it's our niche: mundane things presented as major
achievements, see `larp`):
- **Typography of authority:** book serif for the captions, mono for labels, small caps, numbered points,
  footnote markers, `Fig. 1` labels, a paper or specimen-sheet ground (our `poster` node is already this format).
- **Framing devices:** thesis → evidence → "therefore" structure, citations to nobody ("(Muk, 2026)"), definitions,
  an abstract, a diagram with labelled arrows pointing at an ordinary object.
- **Deadpan delivery:** serious, unhurried, no wink. The content stays absurd or mundane; the form stays formal.
- **Contrast is the joke:** the larger the gap between the gravity of the form and the triviality of the subject,
  the better.

## What wonhee lacked (now designed in the spec)

Word timings in `subtitles`; a timed, animated text and image track in one encode (pop, fade, typewriter,
word highlight, list scroll, insets); bundled fonts with tracking, line height and caption boxes; pause trimming
inside a take; a lofi camcorder look; templates for the five archetypes plus the pseudo-intellectual one; a single
"make a reel in this style" command.
