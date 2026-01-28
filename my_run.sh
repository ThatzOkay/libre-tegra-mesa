#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# vblank_mode=0 disables vsync. Shows max FPS instead of 60

export vblank_mode=0
export GRATE_DEBUG=unimplemented,tgsi,trace
export LIBGL_DEBUG=verbose
export MESA_DEBUG=flush,incomplete_tex,incomplete_fbo,context
export MESA_LOG_LEVEL=debug
export GALLIUM_TRACE=/tmp/trace-gallium.xml
export GALLIUM_TRACE_TC=true
export EGL_LOG_LEVEL=debug
#export EGL_PLATFORM=surfaceless
export NIR_DEBUG=tgsi,print

meson devenv -C build $@
