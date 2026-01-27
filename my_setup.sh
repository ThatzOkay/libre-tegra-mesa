#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source ./my_common.sh

# gnu2 tlsdesc is broken in binutils
export CFLAGS="$CFLAGS -mtls-dialect=gnu"
export CXXFLAGS="$CXXFLAGS -mtls-dialect=gnu"

EXTRA_ARGS=""
if [[ -f "/sys/devices/soc0/family" ]] && [[ "$(cat /sys/devices/soc0/family)" = "Tegra" ]]; then
	echo "Tegra! skipping drm-shim";
else
	EXTRA_ARGS="${EXTRA_ARGS} -Dtools=drm-shim"
fi

meson setup \
	-Dbuildtype=debug \
	-Db_ndebug=false \
	-Dgallium-drivers=grate \
	-Dvulkan-drivers= \
	-Dvideo-codecs= \
	-Dplatforms=wayland,x11 \
	-Dgles1=disabled \
	-Dgles2=enabled \
	-Dgbm=enabled \
	-Dglx=dri \
	-Degl=enabled \
	-Dgallium-va=disabled \
	-Dlibunwind=disabled \
	-Dllvm=disabled \
	-Degl-native-platform=drm \
	-Dprefix="${INSTALL_DIR}" \
	${EXTRA_ARGS} \
	${@} \
	"${BUILD_DIR}"
