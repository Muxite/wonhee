#!/bin/sh
# xvfb-run itself was observed to hang under this image (likely an xauth/lock-file issue);
# starting Xvfb directly and setting DISPLAY works reliably, so do that instead.
set -e
Xvfb :99 -screen 0 16x16x24 >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!
trap 'kill $XVFB_PID 2>/dev/null || true' EXIT
export DISPLAY=:99
sleep 1
exec /usr/local/bin/wavesim_headless "$@"
