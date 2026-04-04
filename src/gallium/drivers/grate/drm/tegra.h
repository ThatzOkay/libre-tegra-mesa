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

#ifndef __DRM_TEGRA_H__
#define __DRM_TEGRA_H__ 1

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "drm-uapi/tegra_drm.h"

enum drm_tegra_class {
    DRM_TEGRA_HOST1X,
    DRM_TEGRA_GR2D,
    DRM_TEGRA_GR3D,
    DRM_TEGRA_VIC,
};

struct grate_bo;
struct grate_device;

struct drm_tegra_channel;
struct drm_tegra_mapping;
struct drm_tegra_pushbuf;
struct drm_tegra_job;
struct drm_tegra_syncpoint;

enum drm_tegra_sync_cond {
    DRM_TEGRA_SYNC_COND_IMMEDIATE,
    DRM_TEGRA_SYNC_COND_OP_DONE,
    DRM_TEGRA_SYNC_COND_RD_DONE,
    DRM_TEGRA_SYNC_COND_WR_SAFE,
    DRM_TEGRA_SYNC_COND_MAX,
};

struct drm_tegra_fence {
    struct grate_device *drm;
    uint32_t syncpt;
    uint32_t value;
};

int drm_tegra_channel_open(struct grate_device *drm,
                           enum drm_tegra_class client,
                           struct drm_tegra_channel **channelp);
int drm_tegra_channel_close(struct drm_tegra_channel *channel);
unsigned int drm_tegra_channel_get_version(struct drm_tegra_channel *channel);
int drm_tegra_channel_map(struct drm_tegra_channel *channel,
                          struct grate_bo *bo, uint32_t flags,
                          struct drm_tegra_mapping **mapp);
int drm_tegra_channel_unmap(struct drm_tegra_mapping *map);

int drm_tegra_job_new(struct drm_tegra_channel *channel,
                      struct drm_tegra_job **jobp);
int drm_tegra_job_free(struct drm_tegra_job *job);
int drm_tegra_job_get_pushbuf(struct drm_tegra_job *job,
                              struct drm_tegra_pushbuf **pushbufp);
int drm_tegra_job_submit(struct drm_tegra_job *job,
                         struct drm_tegra_fence *fence);
int drm_tegra_job_wait(struct drm_tegra_job *job, unsigned long timeout);

int drm_tegra_pushbuf_begin(struct drm_tegra_pushbuf *pushbuf,
                            unsigned int words, uint32_t **ptrp);
int drm_tegra_pushbuf_end(struct drm_tegra_pushbuf *pushbuf, uint32_t *ptr);
int drm_tegra_pushbuf_wait(struct drm_tegra_pushbuf *pushbuf,
                           struct drm_tegra_syncpoint *syncpt,
                           uint32_t value);
int drm_tegra_pushbuf_relocate(struct drm_tegra_pushbuf *pushbuf,
                               uint32_t **ptrp,
                               struct drm_tegra_mapping *target,
                               unsigned long offset, unsigned int shift,
                               uint32_t flags);
int drm_tegra_pushbuf_sync(struct drm_tegra_pushbuf *pushbuf,
                           struct drm_tegra_syncpoint *syncpt,
                           unsigned int count);
int drm_tegra_pushbuf_sync_cond(struct drm_tegra_pushbuf *pushbuf,
                                uint32_t **ptrp,
                                struct drm_tegra_syncpoint *syncpt,
                                enum drm_tegra_sync_cond cond);

int drm_tegra_syncpoint_new(struct grate_device *drm,
                            struct drm_tegra_syncpoint **syncptp);
int drm_tegra_syncpoint_free(struct drm_tegra_syncpoint *syncpt);
int drm_tegra_fence_wait(struct drm_tegra_fence *fence, unsigned long timeout);

#endif /* __DRM_TEGRA_H__ */
