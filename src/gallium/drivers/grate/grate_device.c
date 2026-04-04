/*
 * Copyright © 2012, 2013 Thierry Reding
 * Copyright © 2013 Erik Faye-Lund
 * Copyright © 2014 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include "util/hash_table.h"
#include <util/u_atomic.h>
#include <util/macros.h>

#include <xf86drm.h>

#include <pthread.h>
#include "drm-uapi/tegra_drm.h"

#include "tegra.h"
#include "grate_device.h"
#include "grate_bo.h"

static enum drm_tegra_soc_id read_chip_id(const char *path)
{
	FILE *file = fopen(path, "r");
	if (file) {
		unsigned int id = 0;

		if (fscanf(file, "%d", &id) != 1)
			fprintf(stderr, "fscanf failed for %s\n", path);
		fclose(file);

		switch (id) {
		case 0x20:
			return DRM_TEGRA_SOC_T20;
		case 0x30:
			return DRM_TEGRA_SOC_T30;
		case 0x35:
			return DRM_TEGRA_SOC_T114;
		}

		return DRM_TEGRA_SOC_UNKNOWN;
	} else if (getenv("GRATE_SOC")) {
		// Fix me: my brain is fried :)
		char *str = getenv("GRATE_SOC");
		printf("forced SoC: %s\n", str);
		return DRM_TEGRA_SOC_T30;
	}

	return DRM_TEGRA_SOC_INVALID;
}

enum drm_tegra_soc_id drm_tegra_get_soc_id(void)
{
	static enum drm_tegra_soc_id sid = DRM_TEGRA_SOC_INVALID;

	if (sid != DRM_TEGRA_SOC_INVALID)
		return sid;

	sid = read_chip_id("/sys/devices/soc0/soc_id");
	if (sid != DRM_TEGRA_SOC_INVALID)
		return sid;

	return DRM_TEGRA_SOC_UNKNOWN;
}

static void grate_device_setup_debug(struct grate_device *drm)
{
#ifndef NDEBUG
    char *str;

    str = getenv("GRATE_DEBUG_BO");
    drm->debug_bo = (str && strcmp(str, "1") == 0);
#else 
    
#endif
}

static int grate_device_wrap(struct grate_device **drmp, int fd, bool close)
{
    struct grate_device *drm;
    struct drm_tegra_channel *drm_channel;
    int err;

    if (fd < 0 || !drmp)
        return -EINVAL;

    drm = calloc(1, sizeof(*drm));
    if (!drm)
        return -ENOMEM;

    drm->close = close;
    drm->fd = fd;
    drm->soc_id = drm_tegra_get_soc_id();
    
	grate_device_setup_debug(drm);
	
    err = drm_tegra_channel_open(drm, DRM_TEGRA_GR3D, &drm_channel);
    if (err) {
       fprintf(stderr, "drm_tegra_channel_open err: %d\n", err);
       grate_device_close(drm);
       return err;
    }

    drm_tegra_channel_close(drm_channel);
    
    util_sparse_array_init(&drm->bo_map, sizeof(struct grate_bo), 512);
    pthread_mutex_init(&drm->bo_map_lock, NULL);
    
    *drmp = drm;

    return 0;
}

int grate_device_new(int fd, struct grate_device **drmp)
{
    bool supported = false;
    drmVersionPtr version;

    version = drmGetVersion(fd);
    if (!version) {
       fprintf(stderr, "cannot get version: %s", strerror(errno));
       return -ENOMEM;
    }

    if (!strncmp(version->name, "tegra", version->name_len)) {
        supported = true;
    } else {
       fprintf(stderr, "%s drmGetVersion name got: '%s'\n", __func__, version->name);
    }

    drmFreeVersion(version);

    if (!supported) {
        return -ENOTSUP;
    }

    return grate_device_wrap(drmp, fd, false);
}

void grate_device_close(struct grate_device *drm)
{
    if (!drm)
        return;

    if (drm->close)
        close(drm->fd);
    
    util_sparse_array_finish(&drm->bo_map);
    pthread_mutex_destroy(&drm->bo_map_lock);

    free(drm);
}

