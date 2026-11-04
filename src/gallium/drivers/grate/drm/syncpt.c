/*
 * Copyright © 2021 NVIDIA Corporation
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
#include "tegra_private.h"
#include "tegra_classic.h"

/*
 * The classic kernel interface hides sync points: the "syncpoint id" of a channel is its context id (GET_SYNCPT
 * returns it), and the fence a submit returns is a per-context job counter that SYNCPT_WAIT understands.
 */
int
drm_tegra_syncpoint_new(struct drm_tegra_channel *channel,
                        struct drm_tegra_syncpoint **syncptp)
{
    struct grate_classic_get_syncpt args;
    struct drm_tegra_syncpoint *syncpt;
    int err;

    syncpt = calloc(1, sizeof(*syncpt));
    if (!syncpt)
        return -ENOMEM;

    memset(&args, 0, sizeof(args));
    args.context = channel->context;

    err = ioctl(channel->drm->fd, GRATE_CLASSIC_IOCTL_GET_SYNCPT, &args);
    if (err < 0) {
        free(syncpt);
        return -errno;
    }

    syncpt->drm = channel->drm;
    syncpt->id = args.id;

    *syncptp = syncpt;

    return 0;
}

int
drm_tegra_syncpoint_free(struct drm_tegra_syncpoint *syncpt)
{
    if (!syncpt)
        return -EINVAL;

    free(syncpt);

    return 0;
}

/* timeout is in nanoseconds relative to now; the classic ioctl takes milliseconds */
int
drm_tegra_fence_wait(struct drm_tegra_fence *fence, unsigned long timeout)
{
    struct grate_classic_syncpt_wait args;
    int err;

    memset(&args, 0, sizeof(args));
    args.id = fence->syncpt;
    args.thresh = fence->value;
    args.timeout = timeout / 1000000;

    err = ioctl(fence->drm->fd, GRATE_CLASSIC_IOCTL_SYNCPT_WAIT, &args);
    if (err < 0)
        return -errno;

    return 0;
}
