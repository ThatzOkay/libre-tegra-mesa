#!/bin/sh

#export LD_PRELOAD=build/src/grate/drm-shim/libgrate_noop_drm_shim.so
#export LD_LIBRARY_PATH=build/src/glx/libGL.so

#glxgears -info
#EGL_PLATFORM=surfaceless \

if [[ -z "${SHUT_DEBUG}" ]]; then
	export vblank_mode=0
	export GRATE_DEBUG=unimplemented,tgsi,trace
	export LIBGL_DEBUG=verbose
	export MESA_DEBUG=flush,incomplete_tex,incomplete_fbo,context
	export MESA_LOG_LEVEL=debug
	export GALLIUM_TRACE=/tmp/trace-gallium.xml
	export GALLIUM_TRACE_TC=true
	export EGL_LOG_LEVEL=debug
	export EGL_PLATFORM=surfaceless
	export LIBDRM_TEGRA_DEBUG_BO=1
	export DRI_PRIME_DEBUG=1
	export DRM_SHIM_DEBUG=1
fi

export GRATE_SOC=T30
INSTALL_PATH="${PWD}/build/install_dir/lib/$(uname -m)-linux-gnu"

if [[ ! -d "${INSTALL_PATH}/dri" ]]; then
	INSTALL_PATH="${PWD}/build/install_dir/lib"
fi
if [[ ! -d "${INSTALL_PATH}/dri" ]]; then
	echo "Install path not found! '${INSTALL_PATH}'"
	exit 1
fi
if [[ -z "${1}" ]]; then
	set eglgears_wayland -info
fi

echo "Launching: ${@}"
LD_LIBRARY_PATH="${INSTALL_PATH}" \
	LIBGL_DRIVERS_PATH="${INSTALL_PATH}/dri" \
	LD_PRELOAD="${PWD}/build/src/gallium/drivers/grate/drm-shim/libtegra_noop_drm_shim.so" \
	MESA_LOADER_DRIVER_OVERRIDE=tegra \
	"${@}"
