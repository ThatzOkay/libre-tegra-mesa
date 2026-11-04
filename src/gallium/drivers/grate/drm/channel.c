/*
 * Copyright © 2012, 2013 Thierry Reding
 * Copyright © 2013 Erik Faye-Lund
 * Copyright © 2014-2021 NVIDIA Corporation
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

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <errno.h>
#include <string.h>

#include <sys/ioctl.h>

#include "grate_device.h"
#include "grate_bo.h"
#include "tegra_private.h"
#include "tegra_classic.h"

int
drm_tegra_channel_open(struct grate_device *drm,
                       enum drm_tegra_class client,
                       struct drm_tegra_channel **channelp)
{
    struct grate_classic_open_channel args;
    struct drm_tegra_channel *channel;
    enum host1x_class class;
    int err;

    switch (client) {
    case DRM_TEGRA_HOST1X:
        class = HOST1X_CLASS_HOST1X;
        break;

    case DRM_TEGRA_GR2D:
        class = HOST1X_CLASS_GR2D;
        break;

    case DRM_TEGRA_GR3D:
        class = HOST1X_CLASS_GR3D;
        break;

    case DRM_TEGRA_VIC:
        class = HOST1X_CLASS_VIC;
        break;

    default:
        return -EINVAL;
    }

    channel = calloc(1, sizeof(*channel));
    if (!channel)
        return -ENOMEM;

    channel->drm = drm;

    memset(&args, 0, sizeof(args));
    args.client = class;

    err = ioctl(drm->fd, GRATE_CLASSIC_IOCTL_OPEN_CHANNEL, &args);
    if (err < 0) {
        free(channel);
        return -errno;
    }

    channel->context = args.context;
    channel->capabilities = 0;
    channel->class = class;

    /* the classic interface does not report the hardware version: take it from the SoC */
    switch (drm->soc_id) {
    case DRM_TEGRA_SOC_T20:
        channel->version = 0x20;
        break;

    case DRM_TEGRA_SOC_T30:
        channel->version = 0x30;
        break;

    default:
        channel->version = 0;
        break;
    }

    switch (channel->version) {
    case 0x20:
    case 0x30:
    case 0x35:
    case 0x40:
    case 0x21:
        channel->cond_shift = 8;
        break;

    case 0x18:
    case 0x19:
        channel->cond_shift = 10;
        break;

    default:
        free(channel);
        return -ENOTSUP;
    }

    util_dynarray_init(&channel->pending_bos, NULL);
    *channelp = channel;

    return 0;
}

/* Complete all submitted work before presentation, channel close, or a bounded
 * queue rollover. Retaining BOs prevents GEM handle reuse while jobs are queued.
 */
int drm_tegra_channel_drain(struct drm_tegra_channel *channel)
{
    if (!channel->pending_jobs)
        return 0;
    int err = drm_tegra_fence_wait(&channel->pending, 1000000000UL);
    if (err)
        return err;
    util_dynarray_foreach(&channel->pending_bos, struct grate_bo *, item) {
        struct grate_bo *bo = *item;
        if (bo->pending_channel == channel)
            bo->pending_channel = NULL;
        grate_bo_unref(bo);
    }
    util_dynarray_clear(&channel->pending_bos);
    channel->pending_jobs = 0;
    return 0;
}

int drm_tegra_channel_close(struct drm_tegra_channel *channel)
{
    struct grate_classic_close_channel args;
    struct grate_device *drm;
    int err;

    if (!channel)
        return -EINVAL;

    err = drm_tegra_channel_drain(channel);
    if (err)
        return err;
    drm = channel->drm;

    memset(&args, 0, sizeof(args));
    args.context = channel->context;

    err = ioctl(drm->fd, GRATE_CLASSIC_IOCTL_CLOSE_CHANNEL, &args);
    if (err < 0)
        return -errno;

    util_dynarray_fini(&channel->pending_bos);
    grate_bo_unref(channel->gather);
    free(channel);

    return 0;
}

unsigned int
drm_tegra_channel_get_version(struct drm_tegra_channel *channel)
{
    return channel->version;
}

/*
 * The classic interface has no mapping objects: the kernel maps a buffer when a job that relocates it is submitted.
 * A "mapping" is therefore just the GEM handle, which drm_tegra_pushbuf_relocate() passes on as the relocation target.
 */
int
drm_tegra_channel_map(struct drm_tegra_channel *channel,
                      struct grate_bo *bo, uint32_t flags,
                      struct drm_tegra_mapping **mapp)
{
    struct drm_tegra_mapping *map;

    if (!channel || !bo || !mapp)
        return -EINVAL;

    map = calloc(1, sizeof(*map));
    if (!map)
        return -ENOMEM;

    map->channel = channel;
    map->id = bo->handle;
    map->flags = flags;
    *mapp = map;

    return 0;
}

int
drm_tegra_channel_unmap(struct drm_tegra_mapping *map)
{
    if (!map)
        return -EINVAL;

    free(map);
    return 0;
}
