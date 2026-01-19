#!/bin/sh

#export LD_PRELOAD=/home/jonas/surface_dev/gpu/mesa/build/src/grate/drm-shim/libgrate_noop_drm_shim.so
#export LD_LIBRARY_PATH=/home/jonas/surface_dev/gpu/mesa/build/src/glx/libGL.so

#glxgears -info
#EGL_PLATFORM=surfaceless \


export vblank_mode=0
export GRATE_DEBUG=unimplemented,tgsi,trace
export LIBGL_DEBUG=verbose
export MESA_DEBUG=flush,incomplete_tex,incomplete_fbo,context
export MESA_LOG_LEVEL=debug
export GALLIUM_TRACE=/tmp/trace-gallium.xml
export GALLIUM_TRACE_TC=true
export EGL_LOG_LEVEL=debug
export EGL_PLATFORM=surfaceless


LIBGL_DRIVERS_PATH=/home/jonas/Desktop/SurfaceRT_DEV/gpu/repos/mesa/install_dir/lib/x86_64-linux-gnu/dri \
	LD_PRELOAD=/home/jonas/Desktop/SurfaceRT_DEV/gpu/repos/mesa/build/src/gallium/drivers/grate/drm-shim/libtegra_noop_drm_shim.so \
	LD_LIBRARY_PATH=/home/jonas/Desktop/SurfaceRT_DEV/gpu/repos/mesa/install_dir/lib/x86_64-linux-gnu \
	LIBGL_DEBUG=verbose \
	MESA_DEBUG=1 \
	DRI_PRIME_DEBUG=1 \
	LIBDRM_TEGRA_DEBUG_BO=1 \
	MESA_LOADER_DRIVER_OVERRIDE=tegra \
	DRM_SHIM_DEBUG=1 \
	GRATE_SOC=T30 \
	eglgears_wayland -info
	
	
	#
