/*
 * Copyright (c) 2016 Dmitry Osipenko <digetx@gmail.com>
 * Copyright (C) 2012-2013 NVIDIA Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *	Arto Merilainen <amerilainen@nvidia.com>
 */

#ifndef GRATE_STREAM_H_
#define GRATE_STREAM_H_

#include <stdint.h>

#include "tegra.h"

#define GRATE_PUSHBUF_WORD(ptr, value) *(ptr)++ = (value)

enum grate_stream_status {
   GRATE_STREAM_FREE,
   GRATE_STREAM_CONSTRUCT,
   GRATE_STREAM_CONSTRUCTION_FAILED,
   GRATE_STREAM_READY,
};

struct grate_stream {
   enum grate_stream_status status;

   struct drm_tegra_job *job;
   struct drm_tegra_channel *channel;
   struct drm_tegra_syncpoint *syncpt;

   struct drm_tegra_pushbuf *pushbuf;

   int num_words;
};

struct grate_reloc {
   const void *addr;
   struct grate_bo *bo;
   uint32_t offset;
   unsigned var_offset;
};

/* Stream operations */
int
grate_stream_create(struct grate_device *drm,
                    struct drm_tegra_channel *channel,
                    struct grate_stream *stream,
                    uint32_t words_num);
void
grate_stream_destroy(struct grate_stream *stream);

int
grate_stream_begin(struct grate_stream *stream, uint32_t **ptrp);

int
grate_stream_end(struct grate_stream *stream, uint32_t **ptrp);

int
grate_stream_flush(struct grate_stream *stream, bool wait);

int
grate_stream_push_reloc(struct grate_stream *stream, uint32_t **ptrp,
                        struct grate_bo *bo, unsigned offset);

int
grate_stream_push_sync_cond(struct grate_stream *stream, uint32_t **ptrp,
                            enum drm_tegra_sync_cond cond);

struct grate_reloc
grate_reloc(const void *var_ptr, struct grate_bo *bo,
            uint32_t offset, uint32_t var_offset);

int
grate_stream_push_words(struct grate_stream *stream, uint32_t **ptrp, const uint32_t *src_buf,
                        unsigned src_words, int num_relocs, ...);

#endif
