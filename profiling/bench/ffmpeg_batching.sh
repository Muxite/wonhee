#!/usr/bin/env bash
# Same edit (trim 1s..end-1s, warm grade, one text overlay, 9:16 1080x1920) done four ways. Prints wall s, cpu s, ssim vs one-pass ref.
set -u
SRC=$1; TAG=$2; D=$(mktemp -d); cd "$D"
cat > t.ass <<'A'
[Script Info]
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BorderStyle, Outline, Shadow, Alignment, MarginV
Style: s,DejaVu Sans,72,&H00FFFFFF,&H00000000,1,4,0,5,0
[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:00.00,0:00:05.00,s,going up Bank St.
A
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC"); END=$(python3 -c "print($DUR-1)")
GRADE="eq=saturation=1.1:gamma_r=1.05:gamma_b=0.95"
X264="-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p"
NV="-c:v h264_nvenc -preset p5 -tune hq -rc vbr -cq 21 -b:v 0 -pix_fmt yuv420p"
t() { local name=$1; shift; /usr/bin/time -f "%e %U %S" -o tm "$@" >/dev/null 2>err || { echo "$name FAILED"; tail -2 err; return; }; read w u s < tm; printf "%s %-26s wall %6.2fs cpu %6.1fs" "$TAG" "$name" "$w" "$(python3 -c "print($u+$s)")"; }
q() { ssim=$(ffmpeg -nostats -i "$1" -i ref.mp4 -lavfi "[0:v][1:v]ssim" -f null - 2>&1 | grep -o "All:[0-9.]*"); echo "  $ssim  $(du -k "$1" | cut -f1)KB"; }
# reference: one pass, x264 slow crf 16 (quality anchor)
ffmpeg -v error -y -ss 1 -to $END -i "$SRC" -vf "$GRADE,ass=t.ass" -c:v libx264 -preset slow -crf 14 -c:a aac ref.mp4
# A: three separate ops, each a full decode+encode (how naive node chains behave)
A() { ffmpeg -v error -y -ss 1 -to $END -i "$SRC" $X264 -c:a aac a1.mp4 && ffmpeg -v error -y -i a1.mp4 -vf "$GRADE" $X264 -c:a copy a2.mp4 && ffmpeg -v error -y -i a2.mp4 -vf ass=t.ass $X264 -c:a copy a3.mp4; }
t "A chain 3 re-encodes x264" bash -c "$(declare -f A); X264='$X264'; GRADE='$GRADE'; SRC='$SRC'; END=$END; A"; q a3.mp4
t "B one graph x264" ffmpeg -v error -y -ss 1 -to $END -i "$SRC" -vf "$GRADE,ass=t.ass" $X264 -c:a aac b.mp4; q b.mp4
t "C one graph nvenc" ffmpeg -v error -y -ss 1 -to $END -i "$SRC" -vf "$GRADE,ass=t.ass" $NV -c:a aac c.mp4; q c.mp4
t "D one graph x264 veryfast" ffmpeg -v error -y -ss 1 -to $END -i "$SRC" -vf "$GRADE,ass=t.ass" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -c:a aac d.mp4; q d.mp4
t "E trim only, stream copy" ffmpeg -v error -y -ss 1 -to $END -i "$SRC" -c copy e.mp4; echo
rm -rf "$D"
