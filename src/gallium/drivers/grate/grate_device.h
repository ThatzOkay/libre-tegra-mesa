#ifndef GRATE_DEVICE_H
#define GRATE_DEVICE_H


#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <util/sparse_array.h>

#ifndef __maybe_unused
#define __maybe_unused  __attribute__((unused))
#endif

enum drm_tegra_soc_id {
	DRM_TEGRA_SOC_INVALID,
	DRM_TEGRA_SOC_UNKNOWN,
	DRM_TEGRA_SOC_T20,
	DRM_TEGRA_SOC_T30,
	DRM_TEGRA_SOC_T114,
};

static __maybe_unused const char * const drm_tegra_soc_names[] = {
	[DRM_TEGRA_SOC_INVALID] = "invalid",
	[DRM_TEGRA_SOC_UNKNOWN] = "unknown",
	[DRM_TEGRA_SOC_T20] = "Tegra20",
	[DRM_TEGRA_SOC_T30] = "Tegra30",
	[DRM_TEGRA_SOC_T114] = "Tegra114",
};

enum drm_tegra_soc_id drm_tegra_get_soc_id(void);

struct grate_device {
    bool close;

    /* Device handle */
    int fd;

#ifndef NDEBUG
    bool debug_bo;
#endif

   pthread_mutex_t bo_map_lock;
   struct util_sparse_array bo_map;
   
   // Identifies the SoC that this GPU belongs to
   enum drm_tegra_soc_id soc_id;
};

int grate_device_new(int fd, struct grate_device **drmp);
void grate_device_close(struct grate_device *drm);

static inline struct grate_bo *
grate_lookup_bo(struct grate_device *drm, uint32_t handle)
{
   return util_sparse_array_get(&drm->bo_map, handle);
}

#endif // GRATE_DEVICE_H