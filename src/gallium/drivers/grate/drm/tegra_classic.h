/*
 * The classic Tegra DRM ioctls (OPEN_CHANNEL, GET_SYNCPT, SUBMIT, SYNCPT_WAIT) as the libre-tegra grate kernel
 * exposes them (include/uapi/drm/grate_drm.h). They are declared here under their own names because they clash
 * with the new-style structures in drm-uapi/tegra_drm.h, which the rest of the driver includes. Sizes and layouts
 * are copied from the kernel header; the ioctl numbers encode the structure size, so they must match exactly.
 */
#ifndef __DRM_TEGRA_CLASSIC_H__
#define __DRM_TEGRA_CLASSIC_H__ 1

#include <stdint.h>
#include <xf86drm.h>

#define GRATE_CLASSIC_SYNCPT_WAIT  0x04
#define GRATE_CLASSIC_OPEN_CHANNEL 0x05
#define GRATE_CLASSIC_CLOSE_CHANNEL 0x06
#define GRATE_CLASSIC_GET_SYNCPT   0x07
#define GRATE_CLASSIC_SUBMIT       0x08

struct grate_classic_syncpt_wait {
    uint32_t id;
    uint32_t thresh;
    uint32_t timeout; /* milliseconds */
    uint32_t value;
};

struct grate_classic_open_channel {
    uint32_t client; /* host1x class */
    uint32_t flags;
    uint64_t context;
};

struct grate_classic_close_channel {
    uint64_t context;
};

struct grate_classic_get_syncpt {
    uint64_t context;
    uint32_t index;
    uint32_t id;
};

struct grate_classic_syncpt {
    uint32_t id;
    uint32_t incrs;
};

struct grate_classic_cmdbuf {
    uint32_t handle;
    uint32_t offset;
    uint32_t words;
    uint32_t pad;
};

struct grate_classic_reloc {
    struct {
        uint32_t handle;
        uint32_t offset;
    } cmdbuf;
    struct {
        uint32_t handle;
        uint32_t offset;
    } target;
    uint32_t shift;
    uint32_t pad;
};

struct grate_classic_submit {
    uint64_t context;
    uint32_t num_syncpts;
    uint32_t num_cmdbufs;
    uint32_t num_relocs;
    uint32_t num_waitchks;
    uint32_t waitchk_mask;
    uint32_t timeout;
    uint64_t syncpts;
    uint64_t cmdbufs;
    uint64_t relocs;
    uint64_t waitchks;
    uint32_t fence;
    uint32_t reserved[5];
};

#define GRATE_CLASSIC_IOCTL_SYNCPT_WAIT DRM_IOWR(DRM_COMMAND_BASE + GRATE_CLASSIC_SYNCPT_WAIT, struct grate_classic_syncpt_wait)
#define GRATE_CLASSIC_IOCTL_OPEN_CHANNEL DRM_IOWR(DRM_COMMAND_BASE + GRATE_CLASSIC_OPEN_CHANNEL, struct grate_classic_open_channel)
#define GRATE_CLASSIC_IOCTL_CLOSE_CHANNEL DRM_IOWR(DRM_COMMAND_BASE + GRATE_CLASSIC_CLOSE_CHANNEL, struct grate_classic_close_channel)
#define GRATE_CLASSIC_IOCTL_GET_SYNCPT DRM_IOWR(DRM_COMMAND_BASE + GRATE_CLASSIC_GET_SYNCPT, struct grate_classic_get_syncpt)
#define GRATE_CLASSIC_IOCTL_SUBMIT DRM_IOWR(DRM_COMMAND_BASE + GRATE_CLASSIC_SUBMIT, struct grate_classic_submit)

#endif /* __DRM_TEGRA_CLASSIC_H__ */
