#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source ./my_common.sh

# gnu2 tlsdesc is broken in binutils
export CFLAGS="$CFLAGS -mtls-dialect=gnu"
export CXXFLAGS="$CXXFLAGS -mtls-dialect=gnu"

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
	-Dtools=drm-shim \
	-Degl-native-platform=drm \
	-Dprefix="${INSTALL_DIR}" \
	${@} \
	"${BUILD_DIR}"
