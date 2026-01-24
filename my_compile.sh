#!/bin/bash

source ./my_common.sh

# gnu2 tlsdesc is broken in binutils
export CFLAGS="$CFLAGS -mtls-dialect=gnu"
export CXXFLAGS="$CXXFLAGS -mtls-dialect=gnu"


time meson compile -C $build_dir

time meson install -C $build_dir
