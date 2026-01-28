#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "drm-shim/drm_shim.h"
#include "drm-uapi/tegra_drm.h"

#include "util/u_math.h"

bool drm_shim_driver_prefers_first_render_node = true;

static int
tegra_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
tegra_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct drm_tegra_gem_create *create = arg;

   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = calloc(1, sizeof(*bo));
   size_t size = ALIGN(create->size, 4096);

   drm_shim_bo_init(bo, size);

   create->handle = drm_shim_bo_get_handle(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static int
tegra_ioctl_gem_mmap(int fd, unsigned long request, void *arg)
{
   struct drm_tegra_gem_mmap *mmap = arg;
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, mmap->handle);

   mmap->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);

   return 0;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_TEGRA_GEM_CREATE] = tegra_ioctl_gem_create,
   [DRM_TEGRA_GEM_MMAP] = tegra_ioctl_gem_mmap,
   [DRM_TEGRA_SYNCPT_READ] = tegra_ioctl_noop,
   [DRM_TEGRA_SYNCPT_INCR] = tegra_ioctl_noop,
   [DRM_TEGRA_SYNCPT_WAIT] = tegra_ioctl_noop,
   [DRM_TEGRA_OPEN_CHANNEL] = tegra_ioctl_noop,
   [DRM_TEGRA_CLOSE_CHANNEL] = tegra_ioctl_noop,
   [DRM_TEGRA_GET_SYNCPT] = tegra_ioctl_noop,
   [DRM_TEGRA_SUBMIT] = tegra_ioctl_noop,
   [DRM_TEGRA_GET_SYNCPT_BASE] = tegra_ioctl_noop,
   [DRM_TEGRA_GEM_SET_TILING] = tegra_ioctl_noop,
   [DRM_TEGRA_GEM_GET_TILING] = tegra_ioctl_noop,
   [DRM_TEGRA_GEM_SET_FLAGS] = tegra_ioctl_noop,
   [DRM_TEGRA_GEM_GET_FLAGS] = tegra_ioctl_noop,
};

void
drm_shim_driver_init(void)
{
   shim_device.bus_type = DRM_BUS_HOST1X;
   shim_device.driver_name = "tegra";
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   drm_shim_override_file(
                          "DRIVER=drm\n"
                          "OF_NAME=host1x\n"
                          "OF_FULLNAME=/host1x@50000000\n"
                          "OF_COMPATIBLE_0=nvidia,tegra20-host1x\n"
                          "OF_COMPATIBLE_N=1\n",
                          "/sys/dev/char/%d:%d/device/uevent", DRM_MAJOR,
                          render_node_minor);
}

