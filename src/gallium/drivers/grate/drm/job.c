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

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <poll.h>

#include "grate_device.h"
#include "grate_bo.h"
#include "tegra_private.h"
#include "tegra_classic.h"

/*
 * GRATE_DRM_TRACE=1 logs every submit and wait to stderr; =2 also dumps the command words of each submit. For finding
 * out whether the driver's jobs reach the kernel and complete (there is no other trace of draws).
 */
int
drm_trace_level(void)
{
    static int level = -1;

    if (level < 0) {
        const char *env = getenv("GRATE_DRM_TRACE");
        level = env ? atoi(env) : 0;
    }

    return level;
}

struct drm_tegra_submit_cmd *
drm_tegra_job_add_command(struct drm_tegra_job *job, uint32_t type,
                          uint32_t flags)
{
    struct drm_tegra_submit_cmd *commands, *command;
    size_t size;

    size = (job->num_commands + 1) * sizeof(*commands);

    commands = realloc(job->commands, size);
    if (!commands)
        return NULL;

    command = &commands[job->num_commands];
    memset(command, 0, sizeof(*command));
    command->type = type;
    command->flags = flags;

    job->commands = commands;
    job->num_commands++;

    return command;
}

int
drm_tegra_job_new(struct drm_tegra_channel *channel,
                  struct drm_tegra_job **jobp)
{
    struct drm_tegra_job *job;

    job = calloc(1, sizeof(*job));
    if (!job)
        return -ENOMEM;

    job->page_size = sysconf(_SC_PAGESIZE);
    job->channel = channel;

    *jobp = job;

    return 0;
}

int drm_tegra_job_free(struct drm_tegra_job *job)
{
    if (!job)
        return -EINVAL;

    if (job->pushbuf)
        drm_tegra_pushbuf_free(job->pushbuf);

    if (job->commands)
        free(job->commands);

    if (job->buffers)
        free(job->buffers);

    free(job);

    return 0;
}

int
drm_tegra_job_get_pushbuf(struct drm_tegra_job *job,
                          struct drm_tegra_pushbuf **pushbufp)
{
    struct drm_tegra_pushbuf *pushbuf;

    if (!job->pushbuf) {
        pushbuf = calloc(1, sizeof(*pushbuf));
        if (!pushbuf)
            return -ENOMEM;

        pushbuf->job = job;

        pushbuf->start = calloc(1, job->page_size);
        if (!pushbuf->start) {
            free(pushbuf);
            return -ENOMEM;
        }

        pushbuf->end = pushbuf->start + job->page_size / 4;
        pushbuf->ptr = pushbuf->start;

        job->pushbuf = pushbuf;
    }

    *pushbufp = job->pushbuf;

    return 0;
}

/*
 * The classic kernel interface wants the command words in a GEM buffer (not a user pointer), one relocation per
 * patched word that names the GEM handle of the target, and the sync point increments listed separately. The
 * driver's new-style bookkeeping maps onto that as: one gather buffer holding all of the pushbuf, one command
 * buffer entry covering it, one relocation per drm_tegra_submit_buf (whose "mapping" is the GEM handle, see
 * drm_tegra_channel_map()) and one sync point entry (the channel's context id, see drm_tegra_syncpoint_new()).
 */
int
drm_tegra_job_submit(struct drm_tegra_job *job, struct drm_tegra_fence *fence)
{
    struct drm_tegra_channel *channel = job->channel;
    struct grate_device *drm = channel->drm;
    struct grate_classic_submit args;
    struct grate_classic_cmdbuf cmdbuf;
    struct grate_classic_syncpt syncpt;
    struct grate_classic_reloc *relocs = NULL;
    struct grate_bo *gather;
    uint32_t handle;
    unsigned int words, i;
    void *map;
    int err;

    if (channel->pending_jobs >= 128) {
        err = drm_tegra_channel_drain(channel);
        if (err)
            return err;
    }
    /* Same-channel submissions execute in order. Across channels (especially
     * gr3d draw -> gr2d clear/copy), finish the previous use first.
     */
    for (i = 0; i < job->num_buffers; i++) {
        struct grate_bo *bo = grate_lookup_bo(drm, job->buffers[i].mapping);
        if (bo->pending_channel && bo->pending_channel != channel) {
            err = grate_bo_wait(bo);
            if (err)
                return err;
        }
    }
    words = job->pushbuf->ptr - job->pushbuf->start;

    /* waiting for another job's sync point is a per-command feature the classic interface does not have */
    for (i = 0; i < job->num_commands; i++) {
        if (job->commands[i].type == DRM_TEGRA_SUBMIT_CMD_WAIT_SYNCPT)
            return -ENOTSUP;
    }

    unsigned int gather_size = align(words * 4, job->page_size);
    if (!channel->gather || channel->gather->size < gather_size) {
        gather = grate_bo_alloc(drm, gather_size, 0);
        if (!gather)
            return -ENOMEM;
        grate_bo_unref(channel->gather);
        channel->gather = gather;
    }
    gather = channel->gather;

    err = grate_bo_map(gather, &map);
    if (err < 0)
        goto out_bo;

    memcpy(map, job->pushbuf->start, words * 4);
    grate_bo_unmap(gather);

    err = grate_bo_get_handle(gather, &handle);
    if (err < 0)
        goto out_bo;

    if (job->num_buffers) {
        relocs = calloc(job->num_buffers, sizeof(*relocs));
        if (!relocs) {
            err = -ENOMEM;
            goto out_bo;
        }
    }

    for (i = 0; i < job->num_buffers; i++) {
        relocs[i].cmdbuf.handle = handle;
        relocs[i].cmdbuf.offset = job->buffers[i].reloc.gather_offset_words * 4;
        relocs[i].target.handle = job->buffers[i].mapping;
        relocs[i].target.offset = job->buffers[i].reloc.target_offset;
        relocs[i].shift = job->buffers[i].reloc.shift;
    }

    memset(&cmdbuf, 0, sizeof(cmdbuf));
    cmdbuf.handle = handle;
    cmdbuf.words = words;

    memset(&syncpt, 0, sizeof(syncpt));
    syncpt.id = job->syncpt.id;
    syncpt.incrs = job->syncpt.increments;

    memset(&args, 0, sizeof(args));
    args.context = channel->context;
    args.num_syncpts = 1;
    args.num_cmdbufs = 1;
    args.num_relocs = job->num_buffers;
    args.syncpts = (uintptr_t)&syncpt;
    args.cmdbufs = (uintptr_t)&cmdbuf;
    args.relocs = (uintptr_t)relocs;

    err = ioctl(drm->fd, GRATE_CLASSIC_IOCTL_SUBMIT, &args) < 0 ? -errno : 0;

    if (drm_trace_level()) {
        fprintf(stderr, "grate-drm: submit class 0x%x: %u words, %u relocs, %u syncpt incrs (syncpt %u) -> %d, fence %u\n",
                channel->class, words, job->num_buffers, job->syncpt.increments, job->syncpt.id, err, args.fence);

        if (drm_trace_level() > 1) {
            for (i = 0; i < job->num_buffers; i++)
                fprintf(stderr, "grate-drm:   reloc %u: word %u -> GEM handle %u, offset %u\n", i,
                        job->buffers[i].reloc.gather_offset_words, job->buffers[i].mapping,
                        (unsigned)job->buffers[i].reloc.target_offset);

            for (i = 0; i < words; i++)
                fprintf(stderr, "grate-drm:   %04x: %08x\n", i * 4, job->pushbuf->start[i]);
        }
    }

    if (err < 0)
        goto out_relocs;

    /*
     * GRATE_DRM_WAIT=<ms> waits for every job right after its submit (the driver itself does not wait for draws) and
     * logs how the wait ends: a timeout means the job never completed, success with a wrong picture means it ran.
     */
    if (getenv("GRATE_DRM_WAIT")) {
        struct grate_classic_syncpt_wait wait;
        int werr;

        memset(&wait, 0, sizeof(wait));
        wait.id = job->syncpt.id;
        wait.thresh = args.fence;
        wait.timeout = atoi(getenv("GRATE_DRM_WAIT"));

        werr = ioctl(drm->fd, GRATE_CLASSIC_IOCTL_SYNCPT_WAIT, &wait) < 0 ? -errno : 0;
        fprintf(stderr, "grate-drm: wait for fence %u of class 0x%x (%u ms) -> %d\n", args.fence, channel->class,
                wait.timeout, werr);
    }

    job->syncpt.fence = args.fence;
    channel->pending = (struct drm_tegra_fence){drm, job->syncpt.id, args.fence};
    channel->pending_jobs++;
    for (i = 0; i < job->num_buffers; i++) {
        struct grate_bo *bo = grate_lookup_bo(drm, job->buffers[i].mapping);
        if (bo->pending_channel != channel) {
            grate_bo_ref(bo);
            util_dynarray_append_typed(&channel->pending_bos, struct grate_bo *, bo);
        }
        bo->pending_channel = channel;
        bo->pending = channel->pending;
    }

    if (fence) {
        fence->drm = drm;
        fence->syncpt = job->syncpt.id;
        fence->value = job->syncpt.fence;
    }

    /*
     * The kernel copies the command words out of the gather buffer during the ioctl (the classic submit path,
     * tegra_drm_copy_and_patch_cmdbufs), so the buffer is not needed once the ioctl has returned.
     */
out_relocs:
    free(relocs);
out_bo:
    return err;
}

/* timeout is in nanoseconds relative to now; the classic ioctl takes milliseconds */
int
drm_tegra_job_wait(struct drm_tegra_job *job, unsigned long timeout)
{
    struct drm_tegra_channel *channel = job->channel;
    struct grate_device *drm = channel->drm;
    struct grate_classic_syncpt_wait args;
    unsigned long ms = timeout / 1000000;
    int err;

    /*
     * The driver passes 1000000 here, which is 1 ms as a relative nanosecond timeout. That is too short to wait for
     * a rendering job on this hardware, so wait at least a second (a deliberate deviation from the original value).
     */
    if (ms < 1000)
        ms = 1000;

    memset(&args, 0, sizeof(args));
    args.id = job->syncpt.id;
    args.thresh = job->syncpt.fence;
    args.timeout = ms;

    unsigned long long t0 = drm_trace_level() ? drm_tegra_now_us() : 0;

    err = ioctl(drm->fd, GRATE_CLASSIC_IOCTL_SYNCPT_WAIT, &args);

    if (drm_trace_level())
        fprintf(stderr, "grate-drm: wait id %u thresh %u (%lu ms) -> %d, blocked %llu us\n", args.id, args.thresh, ms,
                err < 0 ? -errno : 0, drm_tegra_now_us() - t0);

    if (err < 0)
        return -errno;

    return 0;
}
