# wonhee's wavegen image (moved from vizkit 2026-10-19): upstream WaveSim cloned at a pinned
# commit plus wonhee's headless/config patch, applied at build time.
#
# Why a patch rather than a fork: nothing has to leave this machine, the pin is one ARG that
# can be bumped, and the local change is one reviewable file in this repo.
#
# Build (from this directory):  docker build -f wavesim.Dockerfile -t wonhee-wavesim:dev .
#
# Requires an nvidia/cuda devel base (CMake declares LANGUAGES CXX CUDA unconditionally, even
# for the CPU compute path) and --gpus passthrough at `docker run` time; this host has a GPU
# but no host CUDA toolkit, same shape as obscura's "no host Rust toolchain" situation.
#
# The headless/RenderTexture patch avoids needing a real *window*, but SFML's GL context
# creation on Linux still opens an X11 (GLX) connection even for off-screen rendering — found
# by actually running the binary, not visible from source alone. The fix stays fully internal
# to this image: wrap the entrypoint in Xvfb (a virtual X server, never a visible display), so
# nothing about the outside contract changes — still just <params.json> in, one PNG out.

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        patch \
        git \
        ca-certificates \
        pkg-config \
        libx11-dev \
        libxrandr-dev \
        libxcursor-dev \
        libxi-dev \
        libudev-dev \
        libgl1-mesa-dev \
        libfreetype-dev \
        libharfbuzz-dev \
        libflac-dev \
        libvorbis-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ARG WAVESIM_REV=1d78d6360cb0f600b4366ea56073781d7d309e5d
RUN git clone https://github.com/Muxite/WaveSim.git wavesim \
    && cd wavesim && git checkout "$WAVESIM_REV" && git submodule update --init --recursive
COPY 0001-wavesim-headless-visual-params.patch patches/0001-wavesim-headless-visual-params.patch

RUN cd wavesim && patch -p1 --forward < ../patches/0001-wavesim-headless-visual-params.patch

RUN cmake -S wavesim -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target WaveSim -j"$(nproc)"

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 libxrandr2 libxcursor1 libxi6 libudev1 libgl1 libfreetype6 libflac12t64 libvorbis0a \
        xvfb \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/build/WaveSim /usr/local/bin/wavesim_headless
COPY wavesim-entrypoint.sh /usr/local/bin/wavesim-entrypoint.sh
RUN chmod +x /usr/local/bin/wavesim-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/wavesim-entrypoint.sh"]
