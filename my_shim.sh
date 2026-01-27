#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source ./my_common.sh

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
	export NIR_DEBUG=tgsi,print
fi

export GRATE_SOC=T30

LIB_DIR="$(meson introspect --buildoptions build | jq -r '.[] | select(.name=="libdir").value')"
LIB_PATH="${INSTALL_DIR}/${LIB_DIR}"

if [[ -z "${1}" ]]; then
	set eglgears_wayland -info
fi

if [[ ! -d "${LIB_PATH}" ]]; then
       echo "Install path not found! '${LIB_PATH}'. Shim may not use driver libs..."
       exit 1
fi

echo "Launching: ${@}"
LD_LIBRARY_PATH="${LIB_PATH}" \
	LIBGL_DRIVERS_PATH="${LIB_PATH}/dri" \
	LD_PRELOAD="${BUILD_DIR}/src/gallium/drivers/grate/drm-shim/libtegra_noop_drm_shim.so" \
	MESA_LOADER_DRIVER_OVERRIDE=tegra \
	"${@}"
