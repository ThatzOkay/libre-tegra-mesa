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

#ifndef __GRATE_BO_H__
#define __GRATE_BO_H__ 1

#include <stdint.h>
#include <stdlib.h>
#include <util/u_dynarray.h>
#include <util/sparse_array.h>

struct grate_bo {
    struct grate_device *drm;
    uint32_t handle;
    uint64_t offset;
    uint32_t flags;
    uint32_t size;
    int32_t refcnt;
    void *map;
    
    /* List of channel mappings where made for this BO, for cleanup */
    struct util_dynarray channel_maps;
};

struct grate_device;

struct grate_bo *grate_bo_alloc(struct grate_device *drm, uint32_t size, uint32_t flags);
struct grate_bo *grate_bo_ref(struct grate_bo *bo);
void grate_bo_unref(struct grate_bo *bo);
int grate_bo_get_handle(struct grate_bo *bo, uint32_t *handle);
int grate_bo_map(struct grate_bo *bo, void **ptr);
int grate_bo_unmap(struct grate_bo *bo);

int grate_bo_export(struct grate_bo *bo, uint32_t flags);
struct grate_bo* grate_bo_import(struct grate_device *drm, int fd);
#endif /* __GRATE_BO_H__ */
