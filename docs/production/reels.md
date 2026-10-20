# Reels: the six styles and how to make them

Status: built 2026-10-20 to [the reel styles spec](../specs/2026-10-20-reel-styles-design.md) §6 and §7; evidence in
[the reel study](reel-study-2026-10-20.md). A style is a `compose_video` template
(`wonhee/edit/compose_templates/<style>.json`) with slots and default params. Layouts are fractions of a 1080x1920
canvas, taken from the measured positions in the study; every text block is already inside the safe zones.

## The command

```bash
uv run wonhee reel <video> --style <name> [--title "..."] [--cta "..."] [--items "a, b, c"] [--inset img ...] [--set k=v]
uv run wonhee reel styles
```

It runs `speech_subs` (word timings, GPU: take the lock) and then `compose_video(template=<style>)`, and writes
`data/reels/<date>/<name>/<name>.mp4` plus `report.md`. `--title` fills the style's headline, `--cta` its closing line
(a `list` reads it as its last entries), `--items` the list entries, scrapbook chapters (up to 4) or lecture theses,
`--inset` up to 4 images that pop in (scrapbook, series, list). Times (chapters, insets, theses, footnotes, the closing
card) are chosen from the words, on phrase starts. An option a style can't use is listed as a `note:` in the output.
`--set` sets any slot (`wonhee node show compose_video` lists them) and these: `tighten=on|off`, `encoder`, `quality`,
`look=<preset>` (grades the clip, e.g. `lofi_cam`, inside the same single encode), `engine=remote|local` (who writes the lecture),
`name`.

## The styles

| Style | For | Reference trait it reproduces | Slots |
|---|---|---|---|
| `legend` | one long uncut take, a payoff at the end | c48: black canvas, clip window at 22-70%, condensed headline popping in at 1.6 s, serif karaoke captions on a dark box with a lilac `#aa73b2` pill | `title`, `title_t`, `pill`, `box_colour`, `bg` |
| `explainer` | talking head, tips | c49: full-bleed, editorial serif headline for the first 4.7 s, lowercase caption chunks at 72%, pauses trimmed | `title`, `subline`, `title_t`, `cta`, `cta_t` |
| `scrapbook` | explainers with things to show | c50: soft-serif headline on paper, word-by-word captions, numbered marker chapter tiles, popping image insets | `title`, `ch1..4` (+`_t`), `inset1..4` (+`_t`), `cta` |
| `series` | a recurring weekly show | c51: split screen (flat panel or images on top, speaker below), one-word-per-line hook, boxed autocaptions, series title card with a sticker burst (it owns the brand; no pinned header) | `title`, `kicker`, `tag`, `ep`, `hook_t`, `card_t`, `inset1..4`, `cta`, `cta_accent` |
| `list` | voice reads a list | c52: near-black stage, UI title pinned at the top, teleprompter list (active line bright), clip and insets crossfading below | `title`, `items`, `at_ms`, `keywords`, `inset1..4`, `bg` |
| `lecture` | mundane things presented as scholarship | c48's accidental register, done on purpose (study, "The pseudo-intellectual presentation") | `title`, `subtitle`, `fig`, `thesis_1..4`, `footnote_1..3`, `marks`, `cta`, `citation`, `card_t`, `ground`, `ink` |

`lecture` has a paper ground (`--set ground="#14130f" --set ink="#ece6d8"` for a dark slide), the clip in a framed figure
window labelled `Fig. 1` in mono, an editorial title with an italic subtitle, captions as quoted propositions (book
serif, no box, ink, a hanging open quote), theses numbered I., II. that fade in at phrase gaps, footnote markers on
caption words with the footnote in small mono at the bottom of the safe area, and a closing card (`Q.E.D.` and
`(Muk, 2026)`). When you give no title or theses, the `larp` lecture register writes them (local model, $0). The form
is formal and the subject trivial; nothing may claim a real credential, publication or institution. The still form is
the `compose` node with `template: lecture_card` (4:5, for carousels).

## Notes

- Dead air: the talking-head styles (`explainer`, `scrapbook`, `series`, `lecture`) cut pauses of 250 ms or more
  down to 80 ms inside the take; `--set tighten=off` keeps the take whole. `legend` and `list` leave it alone.
- Reference text, names and logos are never copied: default slot text is our own, and only positions, sizes, colours
  and timings come from the study.
- The `explainer` headline and closing line sit on one faint dark box per block, and the `series` closing lines on one darker block,
  because bare thin white type disappeared on bright footage in the live check.
