#ifndef GRATE_COMMON_H
#define GRATE_COMMON_H

#include "grate_screen.h"

#ifndef NDEBUG

#ifdef __GLIBC__
#define grate_msg(fmt, ...) \
   fprintf(stderr, "[%s] %s:%d/%s():\t" fmt, program_invocation_short_name, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else // __GLIBC__
#define grate_msg(fmt, ...) \
   fprintf(stderr, "%s:%d/%s():\t" fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#endif

#define VDBG_DRM(DRM, ...) do {					\
	if ((DRM)->debug_bo) { \
		grate_msg(__VA_ARGS__); \
	} \
} while (0)

#define VDBG_BO(BO, FMT, ...) do { \
	if (BO->drm->debug_bo) \
		grate_msg( \
			"BO %p size %u handle %u " \
			"flags 0x%08X refcnt %d map %p "	FMT, \
			BO, BO->size, BO->handle, \
			BO->flags, p_atomic_read(&BO->refcnt), BO->map, \
			##__VA_ARGS__); \
} while (0)

#else //NDEBUG

#define grate_msg(...) fprintf(stderr, __VA_ARGS__)
#define VDBG_DRM(DRM, FMT, ...)	do {} while (0)
#define VDBG_BO(BO, FMT, ...)	do {} while (0)

#endif // NDEBUG

#define grate_unimplemented() do { \
   if (grate_debug & GRATE_DEBUG_UNIMPLEMENTED) \
      printf("GRATE TODO: %s()\n", __func__); \
} while (0)

#define grate_trace() do { \
   if (grate_debug & GRATE_DEBUG_TRACE) \
      printf("GRATE: %s()\n", __func__); \
} while (0)

#define TGR3D_VAL(reg_name, field_name, value) \
   (((value) << TGR3D_ ## reg_name ## _ ## field_name ## __SHIFT) & \
           TGR3D_ ## reg_name ## _ ## field_name ## __MASK)

#define TGR3D_BOOL(reg_name, field_name, bool) \
   ((bool) ? TGR3D_ ## reg_name ## _ ## field_name : 0)

#define TGR3D_MAX_RENDER_TARGETS 16

#endif // GRATE_COMMON_H
