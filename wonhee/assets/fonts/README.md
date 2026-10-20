Fonts bundled so text renders the same everywhere (host, containers). Both are SIL OFL 1.1; licences alongside.
- NotoSans-Bold.ttf: outline and band presets
- DancingScript.ttf: script preset (variable font; the renderer uses the Bold instance)
Colour emoji (Noto Color Emoji) and CJK (Noto Sans CJK) come from the system fonts; `wonhee doctor` checks them.
- NotoSans-Regular.ttf, NotoSansMono-Regular.ttf: the spec-sheet poster (title and body); same OFL as NotoSans-Bold

Reel-style fonts (docs/specs/2026-10-20-reel-styles-design.md §4), all SIL OFL 1.1, licence beside each as OFL-<Family>.txt.
`wonhee.edit.text.ROLES` maps role names (headline, transcript, editorial, editorial_italic, grotesk, ui, caption,
caption_round, soft_serif, marker, marker_alt, hand, autocaption, pixel, sans, mono, script) to these files; variable
fonts (Inter, Inter Tight, Figtree, Fraunces, Arimo) are used at a named instance or fixed axes. Barlow Condensed
SemiBold, Tinos Bold, Instrument Serif (Regular, Italic), Poppins SemiBold, Caveat Brush, Gochi Hand, Mansalva,
Press Start 2P are static. OFL-Tinos.txt is the OFL text with Tinos's copyright line (google/fonts ships none for it).
