#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source ./my_common.sh

# gnu2 tlsdesc is broken in binutils
export CFLAGS="$CFLAGS -mtls-dialect=gnu"
export CXXFLAGS="$CXXFLAGS -mtls-dialect=gnu"


time meson compile -C $BUILD_DIR

time meson install -C $BUILD_DIR
