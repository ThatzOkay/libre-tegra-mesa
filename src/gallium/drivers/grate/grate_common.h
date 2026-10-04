#ifndef GRATE_COMMON_H
#define GRATE_COMMON_H

#include "util/log.h"
#include "grate_screen.h"

#ifndef NDEBUG

#ifdef __GLIBC__
#define grate_logd(fmt, ...) \
   mesa_logd("[%s] %s:%d/%s():\t" fmt, program_invocation_short_name, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define grate_loge(fmt, ...) \
   mesa_loge("[%s] %s:%d/%s():\t" fmt, program_invocation_short_name, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else // __GLIBC__
#define grate_logd(fmt, ...) \
   mesa_logd("%s:%d/%s():\t" fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define grate_loge(fmt, ...) \
   mesa_loge("%s:%d/%s():\t" fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#endif

#define grate_bo_log(LOGFN, BO, FMT, ...) do { \
	if (BO->drm->debug_bo) \
		LOGFN(FMT \
			" - BO %p size %u handle %u " \
			"flags 0x%08X refcnt %d map %p", \
			##__VA_ARGS__, \
			BO, BO->size, BO->handle, \
			BO->flags, p_atomic_read(&BO->refcnt), BO->map \
		); \
} while (0)

#else //NDEBUG

#define grate_logd(...) mesa_logd(__VA_ARGS__)
#define grate_loge(...) mesa_loge(__VA_ARGS__)
#define grate_bo_log(BO, FMT, ...)	do {} while (0)

#endif // NDEBUG

#define grate_bo_logd(...) grate_bo_log(grate_logd, __VA_ARGS__)
#define grate_bo_loge(...) grate_bo_log(grate_loge, __VA_ARGS__)

#define grate_unimplemented() do { \
   if (grate_debug & GRATE_DEBUG_UNIMPLEMENTED) \
      mesa_logw("GRATE TODO: %s()\n", __func__); \
} while (0)

#define grate_trace() do { \
   if (grate_debug & GRATE_DEBUG_TRACE) \
      mesa_logd("GRATE: %s()\n", __func__); \
} while (0)

#endif // GRATE_COMMON_H
