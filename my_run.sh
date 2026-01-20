#!/bin/sh

# DISPLAY=:0 put app to display, needed for SSH
# vblank_mode=0 disables vsync. Shows max FPS instead of 60
# LIBGL_DRI3_DISABLE=true disable DRI3, tegra-xf86 only implements DRI2. Should be replaced with wayland anyway


export vblank_mode=0
export GRATE_DEBUG=unimplemented,tgsi,trace
export LIBGL_DEBUG=verbose
export MESA_DEBUG=flush,incomplete_tex,incomplete_fbo,context
export MESA_LOG_LEVEL=debug
export GALLIUM_TRACE=/tmp/trace-gallium.xml
export GALLIUM_TRACE_TC=true
export EGL_LOG_LEVEL=debug
export EGL_PLATFORM=surfaceless

LD_PRELOAD=build/src/gallium/drivers/grate/drm-shim/libtegra_noop_drm_shim.so \
DISPLAY=:0 \
vblank_mode=0 \
LIBGL_DEBUG=verbose \
meson devenv -C build $@


#LIBGL_DRI3_DISABLE=true \
