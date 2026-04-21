/*
 * Copyright (c) 2016-2017 Dmitry Osipenko <digetx@gmail.com>
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
 * THE SOFTWARE IS PROVIDED "AS IS\n", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Arto Merilainen <amerilainen@nvidia.com>
 */

#include "drm-uapi/tegra_drm.h"
#include "util/set.h"
#include "util/u_dynarray.h"
#include <linux/errno.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <assert.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "host1x01_hardware.h"
#include "hw_host1x01_uclass.h"
#include "tegra_private.h"
#include "tegra.h"
#include "grate_common.h"
#include "grate_device.h"
#include "grate_bo.h"
#include "grate_stream.h"

/*
 * grate_stream_create(channel)
 *
 * Create a stream for given channel. This function preallocates several
 * command buffers for later usage to improve performance. Streams are
 * used for generating command buffers opcode by opcode using
 * grate_stream_push().
 */

int
grate_stream_create(struct grate_device *drm,
                    struct drm_tegra_channel *channel,
                    struct grate_stream *stream,
                    uint32_t words_num)
{
   int ret = 0;
   stream->status    = GRATE_STREAM_FREE;
   stream->channel   = channel;
   stream->num_words = words_num;
   
   ret = drm_tegra_syncpoint_new(stream->channel->drm, &stream->syncpt);
   if (ret < 0) {
      fprintf(stderr, "%s: drm_tegra_syncpoint_new() failed %d\n", __func__, ret);
      return ret;
   }

   return ret;
}

/*
 * grate_stream_destroy(stream)
 *
 * Destroy the given stream object. All resrouces are released.
 */

void
grate_stream_destroy(struct grate_stream *stream)
{
   if (!stream)
      return;

   drm_tegra_job_free(stream->job);
   drm_tegra_syncpoint_free(stream->syncpt);
}

/*
 * grate_stream_flush(stream, fence)
 *
 * Send the current contents of stream buffer. The stream must be
 * synchronized correctly (we cannot send partial streams). If
 * pointer to fence is given, the fence will contain the syncpoint value
 * that is reached when operations in the buffer are finished.
 */

int
grate_stream_flush(struct grate_stream *stream)
{
   int result = 0;

   if (!stream)
      return -1;

   /* Reflushing is fine */
   if (stream->status == GRATE_STREAM_FREE)
      return 0;

   /* Return error if stream is constructed badly */
   if (stream->status != GRATE_STREAM_READY) {
      result = -1;
      goto cleanup;
   }

   result = drm_tegra_job_submit(stream->job, NULL);
   if (result != 0) {
      grate_msg("drm_tegra_job_submit() failed %d\n", result);
      result = -1;
      goto cleanup;
   }

   /*
   result = drm_tegra_job_wait(stream->job, 1000000);
   if (result != 0) {
      grate_msg("drm_tegra_job_wait() failed %d\n", result);
      result = -1;
   }
   */

cleanup:
   assert(result == 0);

   drm_tegra_job_free(stream->job);

   stream->job = NULL;
   stream->status = GRATE_STREAM_FREE;

   return result;
}

/*
 * grate_stream_begin(stream)
 *
 * This function verifies that the current buffer has enough room for holding
 * the whole stream (this is computed using num_words and num_relocs). The
 * function blocks until the stream buffer is ready for use.
 */

int
grate_stream_begin(struct grate_stream *stream, uint32_t **ptrp)
{
   int ret;

   /* check stream and its state */
   if (!(stream && stream->status == GRATE_STREAM_FREE)) {
      grate_msg("Stream status isn't FREE\n");
      return -1;
   }

   ret = drm_tegra_job_new(stream->channel, &stream->job);
   if (ret != 0) {
      grate_msg("drm_tegra_job_new() failed %d\n", ret);
      return -1;
   }

   ret = drm_tegra_job_get_pushbuf(stream->job, &stream->pushbuf);
   if (ret != 0) {
      grate_msg("drm_tegra_job_get_pushbuf() failed %d\n", ret);
      drm_tegra_job_free(stream->job);
      return -1;
   }

   ret = drm_tegra_pushbuf_begin(stream->pushbuf, stream->num_words, ptrp);
   if (ret != 0) {
      grate_msg("drm_tegra_pushbuf_prepare() failed %d\n", ret);
      drm_tegra_job_free(stream->job);
      return -1;
   }
   
   /*
   ret = drm_tegra_pushbuf_sync_cond(
      stream->pushbuf,
      &stream->ptr,
      stream->syncpt,
      DRM_TEGRA_SYNC_COND_OP_DONE
   );
   if (ret < 0) {
      grate_msg("drm_tegra_pushbuf_sync_cond() failed: %d\n", -ret);
      return 1;
   }
   */

   stream->status = GRATE_STREAM_CONSTRUCT;

   return 0;
}

static int
__grate_stream_get_channel_mapping(struct grate_stream *stream,
                                   struct grate_bo *bo,
                                   uint32_t flags,
                                   struct drm_tegra_mapping **mapping)
{
   struct drm_tegra_mapping* bo_mapping = NULL;
   int ret = 0;
   if (!stream || !bo || !mapping) {
       assert(0);
       return -EINVAL;
   }
   assert((flags & !DRM_TEGRA_CHANNEL_MAP_READ_WRITE) == 0);
   assert((flags & DRM_TEGRA_CHANNEL_MAP_READ_WRITE) != 0);

   util_dynarray_foreach(&bo->channel_maps, struct drm_tegra_mapping *, mapping) {
      if ((*mapping)->channel == stream->channel) {
         uint32_t mapping_flags = (*mapping)->flags;
         if (mapping_flags == flags) {
            bo_mapping = *mapping;
         }
      }
   }
  
   if (!bo_mapping) {
      ret = drm_tegra_channel_map(stream->channel, bo, flags, &bo_mapping);
      if (ret < 0) {
         stream->status = GRATE_STREAM_CONSTRUCTION_FAILED;
         grate_msg("drm_tegra_channel_map() failed: %d\n", ret);
         return ret;
      }
      
      util_dynarray_append(&bo->channel_maps, struct drm_tegra_mapping *, bo_mapping);
   }
   
   if (ret == 0)
      *mapping = bo_mapping;
   
   return ret;
}

/*
 * grate_stream_push_reloc(stream, h, offset)
 *
 * Push a memory reference to the stream.
 */

int
grate_stream_push_reloc(struct grate_stream *stream, uint32_t **ptrp,
                        struct grate_bo *bo,
                        unsigned offset)
{
   int ret;
   struct drm_tegra_mapping *mapping = NULL;

   if (!(stream && stream->status == GRATE_STREAM_CONSTRUCT)) {
      grate_msg("Stream status isn't CONSTRUCT\n");
      return -1;
   }
   
   ret = __grate_stream_get_channel_mapping(stream, bo, DRM_TEGRA_CHANNEL_MAP_READ_WRITE, &mapping);
   if (ret < 0)
      return ret;

   ret = drm_tegra_pushbuf_relocate(stream->pushbuf, ptrp, mapping, 
                                    offset, 0, 0);
   if (ret < 0) {
      stream->status = GRATE_STREAM_CONSTRUCTION_FAILED;
      grate_msg("drm_tegra_pushbuf_relocate() failed %d\n", ret);
      return ret;
   }

   return ret;
}

/*
 * grate_stream_end(stream)
 *
 * Mark end of stream. This function pushes last syncpoint increment for
 * marking end of stream.
 */
 
int
grate_stream_end(struct grate_stream *stream, uint32_t **ptrp)
{
   int ret;

   if (!(stream && stream->status == GRATE_STREAM_CONSTRUCT)) {
      grate_msg("Stream status isn't CONSTRUCT\n");
      return -1;
   }
   
   ret = drm_tegra_pushbuf_sync_cond(
      stream->pushbuf,
      ptrp,
      stream->syncpt,
      DRM_TEGRA_SYNC_COND_IMMEDIATE
   );
   if (ret < 0) {
      grate_msg("drm_tegra_pushbuf_sync_cond() failed: %d\n", -ret);
      return 1;
   }
   
   if (*ptrp >= stream->pushbuf->end) {
      grate_msg("pushbuf ptr overflow %p pushbuf end %p\n", *ptrp, stream->pushbuf->end);
      return 1;
   }
   assert(stream->pushbuf->start < *ptrp);

   ret = drm_tegra_pushbuf_end(stream->pushbuf, *ptrp);
   *ptrp = 0;
   if (ret != 0) {
      stream->status = GRATE_STREAM_CONSTRUCTION_FAILED;
      grate_msg("drm_tegra_pushbuf_sync() failed %d\n", ret);
      return -1;
   }

   stream->status = GRATE_STREAM_READY;

   return 0;
}

/*
 * grate_reloc (variable, handle, offset)
 *
 * This function creates a reloc allocation. The function should be used in
 * conjunction with grate_stream_push_words.
 */

struct grate_reloc
grate_reloc(const void *var_ptr, struct grate_bo *bo,
            uint32_t offset, uint32_t var_offset)
{
   struct grate_reloc reloc = {var_ptr, bo, offset, var_offset};
   return reloc;
}

/*
 * grate_stream_push_words(stream, addr, words, ...)
 *
 * Push words from given address to stream. The function takes
 * reloc structs as its argument. You can generate the structs with grate_reloc
 * function.
 */

int grate_stream_push_words(struct grate_stream *stream, uint32_t **ptrp, const uint32_t *src_buf,
                            unsigned src_words, int num_relocs, ...)
{
   struct grate_reloc reloc_arg;
   struct drm_tegra_mapping *mapping;
   va_list ap;
   uint32_t *pushbuf_ptr;
   int ret = 0;

   if (!(stream && stream->status == GRATE_STREAM_CONSTRUCT)) {
      grate_msg("Stream status isn't CONSTRUCT\n");
      return -1;
   }

   /* Copy the contents */
   memcpy(*ptrp, src_buf, src_words * sizeof(uint32_t));

   /* Copy relocs */
   va_start(ap, num_relocs);
   for (; num_relocs; num_relocs--) {
      reloc_arg = va_arg(ap, struct grate_reloc);

      pushbuf_ptr = *ptrp + (reloc_arg.var_offset / sizeof(uint32_t));

      mapping = NULL;
      ret = __grate_stream_get_channel_mapping(stream, reloc_arg.bo, DRM_TEGRA_CHANNEL_MAP_READ_WRITE, &mapping);
      if (ret < 0)
         break;

      ret = drm_tegra_pushbuf_relocate(stream->pushbuf,
                                       &pushbuf_ptr, mapping,
                                       reloc_arg.offset, 0, 0);
      if (ret != 0) {
         stream->status = GRATE_STREAM_CONSTRUCTION_FAILED;
         grate_msg("__grate_stream_get_channel_mapping() failed %d\n", ret);
         break;
      }
   }
   va_end(ap);

   *ptrp += src_words;

   return ret ? -1 : 0;
}
