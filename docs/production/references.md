# Reference set (2026-10-17)

These were grabbed with `wonhee grab` (runs 10–11). Content ids are wonhee `content_item` ids, and the media stays in `data/`, uncommitted.
Plays and likes are as of 2026-10-17.

## Ours: @frictionlessmuk

| id | type | caption | result |
|---|---|---|---|
| 14 | reel, 12.8 s, 1 shot | "Let's connect???!!" + 5 hashtags; text overlay "waterloo dream setup / @hackTheNorth / claude + agents + starlink = holy startup"; audio "All I Do Is Win" | 3,095 plays, 66 likes |
| 5, 7, 9, 10 | photo / carousel posts | date only ("2025-04-29 Stanley Park Adventure") | 60–122 likes |

## External reels (video downloaded)

| id | account | length / shots | format | plays |
|---|---|---|---|---|
| 15 | sirbrandyyy | 12.8 s / 1 | one shot, POV text overlay ("POV when you're working really hard and remember your salary"), original audio | 3.78M |
| 18 | hntrdi | 20.3 s / 7, **720×720** | music edit (Deftones) | 2.64M |
| 12 | suaheffley | 78.5 s / 2 | spoken skit | 2.33M |
| 17 | y_og.l | 13.7 s / 1 | one shot to a song | 2.09M |
| 4 | ahmedlikespublictransit | 12.9 s / 3 | per-shot script text + emoji, warm grade, trending song | 780k |
| 13 | ism.not.iit | 10.4 s / 5 | quick cuts, no speech | 715k |
| 11 | _bosco0715_ | 10.3 s / 1 | one shot to a trending song | 177k |
| 2 | christine.choiiii | 46.4 s / 12 | talking-head commentary | 102k |
| 19 | magnumwrapper | 10.7 s / 4 | celebrity clip edit with music | 101k |
| 20 | dredbaby.price | 109.7 s / 24 | vlog | (likes 205k) |

External photo / carousel posts (not downloaded, see gaps): 6 `_yujin_an` (297k likes), 8 `eeseooes` (an ad, 162k likes), 16 `_chaechae_1` (1.06M likes).

## What the set says about production

- **The winning format is the cheap one.** 7 of the 11 reels run 10–14 s, and 4 are a **single shot**. A single shot plus a text overlay plus audio is exactly our reel 14's format, which got 3.1k plays, while sirbrandyyy's got 3.8M. Render cost is the same. The gap is the hook (relatable POV text), the audio choice, and account reach, not editing.
- So for this style, **render time isn't the bottleneck**. One NVENC render of a 13 s clip takes 2–3 s. Time goes to choosing the clip, writing the overlay text and caption, and picking audio. That's where workers should help: the shot index (search your footage), caption and overlay drafts in your voice (style memory), and a quick preview.
- Your feed is mostly **photo carousels with date captions**. A carousel worker (`slopcut carousel` already exists) and caption drafts would speed up the most common post you make.
- Square 720×720 (id 18) still did 2.6M plays. 9:16 isn't mandatory for music edits. The check should warn about it, not block it.

## Gaps this batch exposed

- **Photo and carousel posts aren't downloaded** by `grab` (5 of 17 posts: "video: not downloaded"). Add image download for `/p/` posts so carousels can serve as references and style sources.
- **Whisper transcribes song lyrics as speech** in 3 more reels (11, 14, 17), alongside the "Bye." in reel 4. The music gate in watchouts.md is needed.
- **grab and analyze run strictly one post at a time.** 17 posts took 156 s to grab and 131 s to analyze 10 videos (about 13 s each), at a cost of $0.054 in total. Downloads (network) can run about 4 at once. In analyze, shot detection (CPU), whisper (GPU) and the vision call (network) can overlap across videos. The serial loops are in `wonhee/library/grab.py`, both the download loop and `analyze()`.
