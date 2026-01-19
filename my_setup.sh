#!/bin/sh

#-Dlegacy-x11=dri2 is needed for newer mesa

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
	-Dprefix=/home/jonas/Desktop/SurfaceRT_DEV/gpu/repos/mesa/install_dir \
	build


