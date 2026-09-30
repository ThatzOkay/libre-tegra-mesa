#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source ./my_common.sh

#EXTRA_SETUP_ARGS="${EXTRA_SETUP_ARGS} -D b_sanitize=address -D tools=dlclose-skip -D valgrind=disabled"

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
	-Dprefix="${INSTALL_DIR}" \
	${EXTRA_SETUP_ARGS} \
	${@} \
	"${BUILD_DIR}"
