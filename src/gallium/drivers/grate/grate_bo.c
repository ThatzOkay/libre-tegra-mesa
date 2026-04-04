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
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include <util/os_mman.h>
#include <util/u_atomic.h>
#include <util/macros.h>

#include <xf86drm.h>

#include "drm-uapi/tegra_drm.h"

#include "tegra.h"
#include "grate_common.h"
#include "grate_device.h"
#include "grate_bo.h"

static void grate_bo_free(struct grate_bo *bo)
{
    struct grate_device *drm = bo->drm;

    VDBG_BO(bo, "\n");

    if (bo->map)
        munmap(bo->map, bo->size);

    drmCloseBufferHandle(drm->fd, bo->handle);
    
    util_dynarray_foreach(&bo->channel_maps, struct drm_tegra_mapping *, mapping) {
       drm_tegra_channel_unmap(*mapping);
    }
    util_dynarray_fini(&bo->channel_maps);
}

static void grate_bo_prepare(struct grate_bo *bo)
{
   p_atomic_set(&bo->refcnt, 1);
   util_dynarray_init(&bo->channel_maps, NULL);
}

struct grate_bo *grate_bo_alloc(struct grate_device *drm, uint32_t size, uint32_t flags)
{
    struct drm_tegra_gem_create args;
    struct grate_bo *bo;
    int err;

    if (!drm || size == 0)
        return NULL;

    memset(&args, 0, sizeof(args));
    args.flags = flags;
    args.size = size;

    err = drmCommandWriteRead(drm->fd, DRM_TEGRA_GEM_CREATE, &args,
                              sizeof(args));
    if (err < 0) {
        VDBG_DRM(drm, "failed size %u bytes flags 0x%08X err %d (%s)\n",
           size, flags, err, strerror(-err));
        return NULL;
    }
    
    pthread_mutex_lock(&drm->bo_map_lock);
    bo = grate_lookup_bo(drm, args.handle);
    pthread_mutex_unlock(&drm->bo_map_lock);
    
    /* Fresh handle */
    assert(!memcmp(bo, &((struct grate_bo){}), sizeof(*bo)));

    bo->handle = args.handle;
    bo->drm = drm;
    bo->size = args.size;
    bo->flags = flags;
    
    VDBG_BO(bo, "%s", "success new\n");

    return bo;

/*
free:
    grate_bo_free(bo);
    return NULL;
*/
}

struct grate_bo *grate_bo_ref(struct grate_bo *bo)
{
   if (bo) {
      VDBG_BO(bo, "\n");
      ASSERTED int count = p_atomic_inc_return(&bo->refcnt);
      assert(count != 1);
   }

    return bo;
}

void grate_bo_unref(struct grate_bo *bo)
{
   if (!bo)
      return;

   int refcnt = p_atomic_dec_return(&bo->refcnt);
   assert(refcnt >= 0);

   /* Don't return to cache if there are still references */
   if (refcnt)
      return;

   pthread_mutex_lock(&bo->drm->bo_map_lock);

   /* Someone might have imported this BO while we were waiting for the
    * lock, let's make sure it's still not referenced before freeing it.
    */
   if (p_atomic_read(&bo->refcnt) == 0) {
      VDBG_BO(bo, "\n");
      grate_bo_free(bo);
   }
   
   pthread_mutex_unlock(&bo->drm->bo_map_lock);
}

int
grate_bo_get_handle(struct grate_bo *bo, uint32_t *handle)
{
    if (!bo || !handle)
        return -EINVAL;

    *handle = bo->handle;

    return 0;
}

int grate_bo_map(struct grate_bo *bo, void **ptr)
{
    struct grate_device *drm = bo->drm;

    if (!bo->map) {
        struct drm_tegra_gem_mmap args;
        int err;

        memset(&args, 0, sizeof(args));
        args.handle = bo->handle;

        err = drmCommandWriteRead(drm->fd, DRM_TEGRA_GEM_MMAP, &args,
                                  sizeof(args));
        if (err < 0) {
            VDBG_BO(bo, "failed get mapping offset err %d (%s)\n",
               err, strerror(-err));
            return -errno;
        }

        bo->offset = args.offset;

        bo->map = os_mmap(NULL, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                           drm->fd, bo->offset);
        if (bo->map == MAP_FAILED) {
            VDBG_BO(bo, "failed to map offset 0x%llX err %d (%s)\n",
               args.offset, -errno, strerror(errno));
            bo->map = NULL;
            return -errno;
        }

        VDBG_BO(bo, "success\n");
    }

    if (ptr)
        *ptr = bo->map;

    return 0;
}

int grate_bo_unmap(struct grate_bo *bo)
{
    if (!bo)
        return -EINVAL;

    if (!bo->map)
        return 0;

    VDBG_BO(bo, "\n");

    if (munmap(bo->map, bo->size))
        return -errno;

    bo->map = NULL;

    return 0;
}

int grate_bo_export(struct grate_bo *bo, uint32_t flags)
{
    int fd, err;

    flags |= DRM_CLOEXEC;

    err = drmPrimeHandleToFD(bo->drm->fd, bo->handle, flags, &fd);
    if (err < 0) {
        VDBG_BO(bo, "failed err %d strerror(%s)\n",
            err, strerror(-err));
        return err;
    }

    return fd;
}

static ssize_t fd_get_size(int fd)
{
    ssize_t size, offset;
    int err;

    offset = lseek(fd, 0, SEEK_CUR);
    if (offset < 0)
        return -errno;

    size = lseek(fd, 0, SEEK_END);
    if (size < 0)
        return -errno;

    err = lseek(fd, offset, SEEK_SET);
    if (err < 0)
        return -errno;

    return size;
}

struct grate_bo*
grate_bo_import(struct grate_device *drm, int fd)
{
   struct grate_bo *bo;
   ssize_t size;
   int err;
   unsigned gem_handle;

   pthread_mutex_lock(&drm->bo_map_lock);

   err = drmPrimeFDToHandle(drm->fd, fd, &gem_handle);
   if (err < 0) {
      VDBG_DRM(drm, "failed err %d strerror(%s)\n", err, strerror(-err));
      pthread_mutex_unlock(&drm->bo_map_lock);
      return NULL;
   }

   bo = grate_lookup_bo(drm, gem_handle);

   if (!bo->size) {
      size = fd_get_size(fd);
      if (size < 0)
         goto error;
   
      bo->drm = drm;
      bo->size = (uint32_t) size;
      bo->handle = gem_handle;

      grate_bo_prepare(bo);
   } else {
      /* bo->refcnt == 0 can happen if the BO
       * was being released but grate_bo_import() acquired the
       * lock before grate_bo_unref(). In that case, refcnt
       * is 0 and we can't use grate_bo_ref() directly, we
       * have to re-initialize the refcnt().
       * Note that grate_bo_unref() checks
       * refcnt value just after acquiring the lock to
       * make sure the object is not freed if grate_bo_import()
       * acquired it in the meantime.
       */
      if (p_atomic_read(&bo->refcnt) == 0)
         p_atomic_set(&bo->refcnt, 1);
      else
         grate_bo_ref(bo);
   }
   pthread_mutex_unlock(&drm->bo_map_lock);
   
   assert(bo->drm != NULL && "post-condition");

   return bo;
   
error:
   memset(bo, 0, sizeof(*bo));
   pthread_mutex_unlock(&drm->bo_map_lock);
   return NULL;
}
