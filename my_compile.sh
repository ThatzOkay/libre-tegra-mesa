#!/bin/bash

# gnu2 tlsdesc is broken in binutils
export CFLAGS="$CFLAGS -mtls-dialect=gnu"
export CXXFLAGS="$CXXFLAGS -mtls-dialect=gnu"


time meson compile -C build

time meson install -C build
