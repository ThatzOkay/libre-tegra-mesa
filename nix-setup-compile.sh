#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

DEPS="zlib libglvnd glslang libdrm udev llvm spirv-tools lua5_4 libelf valgrind libxcb libx11 libxext libxshmfence libxxf86vm libxrandr libxcb-keysyms libxfixes wayland-protocols wayland-scanner wayland pkg-config"
export CFLAGS="$CFLAGS -O1"
export CXXFLAGS="$CXXFLAGS -O1"
	
if [[ -d build && -z "${FORCE_SETUP}" ]]; then
        echo "> Skipping meson setup, build exists"
else
        echo "> Running meson setup"
        nix-shell -p ${DEPS} --run "./my_setup.sh ${@}"
fi

echo "> Running meson compile"
nix-shell -p ${DEPS} --run "./my_compile.sh"
