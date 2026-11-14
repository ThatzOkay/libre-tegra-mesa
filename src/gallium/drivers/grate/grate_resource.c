#include <stdio.h>

#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_pack_color.h"
#include "util/u_surface.h"
#include "util/u_transfer.h"
#include "util/u_inlines.h"

#include "grate_common.h"
#include "grate_bo.h"
#include "grate_context.h"
#include "grate_resource.h"
#include "grate_screen.h"

#include "host1x01_hardware.h"

/*
 * XXX Required to access winsys_handle internals. Should go away in favour
 * of some abstraction to handle handles in a Tegra-specific winsys
 * implementation.
 */
#include "frontend/drm_driver.h"


static bool
grate_resource_get_handle(struct pipe_screen *pscreen,
                          struct pipe_context *context,
                          struct pipe_resource *presource,
                          struct winsys_handle *handle,
                          unsigned usage)
{
   struct grate_resource *resource;
   int ret;

   if (presource->target == PIPE_BUFFER)
      return false;

   resource = grate_resource(presource);
   if (grate_bo_wait(resource->bo))
      return false;

   if (handle->type == WINSYS_HANDLE_TYPE_KMS) {
      ret = grate_bo_get_handle(resource->bo, &handle->handle);
      if (ret < 0) {
         fprintf(stderr, "grate_bo_get_handle() failed: %d\n", ret);
         return false;
      }
   } else if (handle->type == WINSYS_HANDLE_TYPE_FD) {
      ret = grate_bo_export(resource->bo, 0);
      if (0 < ret) {
         handle->handle = ret;
      } else {
         fprintf(stderr, "grate_bo_export() failed: %d\n", ret);
         return false;
      }
   } else {
      fprintf(stdout, "unsupported handle type: %d\n", handle->type);
      return false;
   }

   handle->stride = resource->pitch;
   return true;
}

static void
grate_resource_destroy(struct pipe_screen *pscreen,
                       struct pipe_resource *presource)
{
   struct grate_resource *resource = grate_resource(presource);

   grate_bo_unref(resource->bo);
   grate_bo_unref(resource->sampler_bo);
   FREE(resource);
}

static void *
grate_resource_transfer_map(struct pipe_context *pcontext,
                            struct pipe_resource *presource,
                            unsigned level, unsigned usage,
                            const struct pipe_box *box,
                            struct pipe_transfer **transfer)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_resource *resource = grate_resource(presource);
   void *ret = NULL;
   struct pipe_transfer *ptrans;

   if (usage & PIPE_MAP_DIRECTLY)
      return NULL;
   if (level > presource->last_level)
      return NULL;

   struct grate_transfer *gtrans = slab_zalloc(&context->transfer_pool);
   if (!gtrans)
      return NULL;
   ptrans = &gtrans->b;

   if (grate_bo_map(resource->bo, &ret)) {
      slab_free(&context->transfer_pool, gtrans);
      return NULL;
   }
   ptrans->stride = resource->level_pitch[level];
   ptrans->layer_stride = ptrans->stride * resource->level_height[level];
   ret = (uint8_t *)ret + resource->level_offset[level] +
         box->z * ptrans->layer_stride + box->y * ptrans->stride +
         box->x * util_format_get_blocksize(presource->format);

   pipe_resource_reference(&ptrans->resource, presource);
   ptrans->level = level;
   ptrans->usage = usage;
   ptrans->box = *box;
   *transfer = ptrans;

   return ret;
}

static void
grate_resource_transfer_flush_region(struct pipe_context *pcontext,
                                     struct pipe_transfer *transfer,
                                     const struct pipe_box *box)
{
   grate_unimplemented();
}

static void
grate_resource_transfer_unmap(struct pipe_context *pcontext,
                              struct pipe_transfer *transfer)
{
   struct grate_context *context = grate_context(pcontext);
   grate_bo_unmap(grate_resource(transfer->resource)->bo);

   pipe_resource_reference(&transfer->resource, NULL);
   slab_free(&context->transfer_pool, transfer);
}

int
grate_pixel_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_A8_UNORM:
      return TGR3D_SURF_FORMAT_A8;
   case PIPE_FORMAT_L8_UNORM:
      return TGR3D_SURF_FORMAT_L8;
   /* Only meant for sampling the planes of an imported planar YUV buffer
    * (I420/YV12 from the video decoder), whose per-plane format is R8.  An L8
    * texel reads (L,L,L,1), so the .r the YUV lowering uses is the plane
    * value.  grate_screen_is_format_supported() keeps R8 out of render
    * targets. */
   case PIPE_FORMAT_R8_UNORM:
      return TGR3D_SURF_FORMAT_L8;
    case PIPE_FORMAT_S8_UINT:
      return TGR3D_SURF_FORMAT_S8;
    case PIPE_FORMAT_L8A8_UNORM:
      return TGR3D_SURF_FORMAT_L8A8;
   case PIPE_FORMAT_B5G6R5_UNORM:
      return TGR3D_SURF_FORMAT_B5G6R5;
   case PIPE_FORMAT_B5G5R5A1_UNORM:
      return TGR3D_SURF_FORMAT_A1B5G5R5;
   case PIPE_FORMAT_B4G4R4A4_UNORM:
      return TGR3D_SURF_FORMAT_A4B4G4R4;
   case PIPE_FORMAT_Z16_UNORM:
      return TGR3D_SURF_FORMAT_Z16;
   /* The Tegra 3 has no 24-bit depth. Like the Windows driver, store D24 as
    * 16-bit non-linear depth (the stencil of Z24S8 lives in a separate S8
    * plane, see grate_resource). */
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      return TGR3D_SURF_FORMAT_Z16NL;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return TGR3D_SURF_FORMAT_R8G8B8A8;
   /* Surface names use the opposite byte order to Gallium's format names. */
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return TGR3D_SURF_FORMAT_B8G8R8A8;
   default:
      return -1;
   }
}

bool
grate_format_is_emulated_z24(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      return true;
   default:
      return false;
   }
}

/* Z16NL is a float-like 16-bit code: 3 exponent bits, 13 mantissa bits. The
 * Windows driver's decode table (it builds one so shaders can read the depth)
 * turns a code into a 20-bit linear value: z20 = ((2^e - 1) << (20 - e)) | (m
 * << max(6 - e, 0)), and the 16-bit linear depth is z20 >> 4. Inverted here:
 * e is the number of leading ones of z20 (at most 7), m the 13 bits after the
 * first zero.
 */
uint16_t
grate_pack_z16nl(double depth)
{
   if (depth <= 0.0)
      return 0;
   if (depth >= 1.0)
      return 0xffff;

   uint32_t z20 = (uint32_t)(depth * 0xfffff + 0.5);
   unsigned e = 0;
   while (e < 7 && (z20 & (1u << (19 - e))))
      e++;
   uint32_t m = (z20 >> (e < 6 ? 6 - e : 0)) & 0x1fff;
   return (e << 13) | m;
}

static struct pipe_resource *
grate_screen_resource_create(struct pipe_screen *pscreen,
                             const struct pipe_resource *template)
{
   struct grate_screen *screen = grate_screen(pscreen);
   struct grate_resource *resource;
   uint32_t flags = 0, height, size;

   resource = CALLOC_STRUCT(grate_resource);
   if (!resource)
      return NULL;

   if (template->last_level >= PIPE_MAX_TEXTURE_LEVELS ||
       (template->last_level &&
        (template->target != PIPE_TEXTURE_2D || template->array_size > 1 ||
         util_format_is_depth_or_stencil(template->format)))) {
      FREE(resource);
      return NULL;
   }

   resource->b = *template;

   pipe_reference_init(&resource->b.reference, 1);
   resource->b.screen = pscreen;

   resource->pitch = template->width0 * util_format_get_blocksize(template->format);
   height = template->height0;

   resource->tiled = 0;
   if (template->bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
                         PIPE_BIND_SCANOUT | PIPE_BIND_DEPTH_STENCIL)) {
      if (template->bind & PIPE_BIND_DEPTH_STENCIL) {
         if (grate_format_is_emulated_z24(template->format))
            resource->pitch = align(template->width0 * 2, 256);
         else
            resource->pitch = align(resource->pitch, 256);
      } else if (template->bind &
                 (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SCANOUT))
         resource->pitch = align(resource->pitch, 32);

      /* The kernel reflects BOTTOM_UP framebuffers at scanout. GBM/KMS
       * buffers already use the display's normal row order. */
      flags = (template->bind & PIPE_BIND_SCANOUT)
                 ? 0
                 : DRM_TEGRA_GEM_CREATE_BOTTOM_UP;
   }

   if (template->target != PIPE_BUFFER) {
      /* pick pixel-format */
      int format = grate_pixel_format(template->format);
      if (format < 0) {
         grate_loge("unsupported resource format %s (target=%u bind=0x%x)\n",
                    util_format_name(template->format), template->target,
                    template->bind);
         FREE(resource);
         return NULL;
      }
      resource->format = format;
   }

   /* Textures use either 16-byte POT rows or 64-byte NPOT rows. Renderable
    * textures can be sampled later, so allocate a native NPOT-compatible
    * stride rather than the framebuffer-only 32-byte alignment.
    */
   if (template->target != PIPE_BUFFER &&
       (template->bind & PIPE_BIND_SAMPLER_VIEW))
      resource->pitch = align(resource->pitch, 64);

   if (template->target == PIPE_TEXTURE_2D &&
       (template->bind & PIPE_BIND_SAMPLER_VIEW) &&
       !(template->bind & PIPE_BIND_DEPTH_STENCIL) &&
       util_is_power_of_two_or_zero(template->width0) &&
       util_is_power_of_two_or_zero(template->height0))
      resource->pitch = align(template->width0 *
                                 util_format_get_blocksize(template->format),
                              16);

   size = resource->pitch * height;
   resource->level_pitch[0] = resource->pitch;
   resource->level_height[0] = height;

   if (template->last_level) {
      unsigned cpp = util_format_get_blocksize(template->format);
      bool pot = util_is_power_of_two_or_zero(template->width0) &&
                 util_is_power_of_two_or_zero(template->height0);
      unsigned alloc_width = util_next_power_of_two(template->width0);
      unsigned alloc_height = util_next_power_of_two(template->height0);

      /* NVIDIA's linear mip chains have 16-byte rows and level extents.
       * NPOT level zero instead uses 64-byte rows, and later levels reserve
       * power-of-two extents. T30 does not align the base extent to 1K.
       */
      resource->pitch = align(template->width0 * cpp, pot ? 16 : 64);
      resource->level_pitch[0] = resource->pitch;
      size = align(resource->pitch * height, 16);
      for (unsigned level = 1; level <= template->last_level; level++) {
         unsigned width = u_minify(template->width0, level);
         unsigned h = u_minify(height, level);
         resource->level_offset[level] = size;
         resource->level_pitch[level] = align(width * cpp, 16);
         resource->level_height[level] = h;
         size += align(align(u_minify(alloc_width, level) * cpp, 16) *
                          u_minify(alloc_height, level),
                       16);
      }
      size = align(size, 1024);
   }

   if (util_format_has_stencil(util_format_description(template->format)) &&
       grate_format_is_emulated_z24(template->format)) {
      resource->stencil_pitch = align(template->width0, 256);
      resource->stencil_offset = size;
      size += resource->stencil_pitch * height;
   }

   resource->bo = grate_bo_alloc(screen->drm, size, flags);
   if (!resource->bo) {
      fprintf(stderr, "grate_bo_alloc() failed\n");
      FREE(resource);
      return NULL;
   }

   return &resource->b;
}

static struct pipe_resource *
grate_screen_resource_from_handle(struct pipe_screen *pscreen,
                                  const struct pipe_resource *template,
                                  struct winsys_handle *handle,
                                  unsigned usage)
{
   struct grate_screen *screen = grate_screen(pscreen);
   struct grate_resource *resource;
   int format;

   /* A dma-buf handle describes one image and supplies no mip offsets. */
   if (template->last_level)
      return NULL;

   resource = CALLOC_STRUCT(grate_resource);
   if (!resource)
      return NULL;

   resource->b = *template;

   pipe_reference_init(&resource->b.reference, 1);
   resource->b.screen = pscreen;

   switch (handle->type) {
   case WINSYS_HANDLE_TYPE_FD:
      resource->bo = grate_bo_import(screen->drm, handle->handle);
      if (!resource->bo) {
         fprintf(stderr, "grate_bo_import() failed\n");
         goto fail;
      }
      break;
   default:
      UNREACHABLE("invalid winsys handle type");
   }
   /* One plane of a multi-plane dma-buf (e.g. the U and V planes of I420) lives
    * at a non-zero offset inside the same buffer. */
   if (!resource->bo || handle->offset >= resource->bo->size)
      goto fail;
   resource->level_offset[0] = handle->offset;

   resource->pitch = handle->stride;
   resource->level_pitch[0] = resource->pitch;
   resource->level_height[0] = template->height0;

   format = grate_pixel_format(template->format);
   if (format < 0) {
      grate_loge("unsupported imported format %s\n",
                 util_format_name(template->format));
      goto fail;
   }
   resource->format = format;

   return &resource->b;
   
fail:
   grate_bo_unref(resource->bo);
   FREE(resource);
   return NULL;
}

void
grate_screen_resource_init(struct pipe_screen *pscreen)
{
   grate_trace();
   pscreen->resource_create = grate_screen_resource_create;
   pscreen->resource_from_handle = grate_screen_resource_from_handle;
   pscreen->resource_get_handle = grate_resource_get_handle;
   pscreen->resource_destroy = grate_resource_destroy;
}

static void
grate_resource_copy_region(struct pipe_context *pcontext,
                           struct pipe_resource *pdst, unsigned int dst_level,
                           unsigned int dstx, unsigned dsty,
                           unsigned int dstz, struct pipe_resource *psrc,
                           unsigned int src_level, const struct pipe_box *box)
{
   struct grate_resource *dst = grate_resource(pdst);
   struct grate_resource *src = grate_resource(psrc);
   void *dst_map, *src_map;

   if (dst_level > pdst->last_level || src_level > psrc->last_level)
      return;
   if (dst->tiled || src->tiled ||
       util_format_get_blocksize(pdst->format) !=
          util_format_get_blocksize(psrc->format)) {
      grate_unimplemented();
      return;
   }

   /* the mapping waits for the GPU */
   if (grate_bo_map(src->bo, &src_map) || grate_bo_map(dst->bo, &dst_map))
      return;

   util_copy_box((uint8_t *)dst_map + dst->level_offset[dst_level],
                 pdst->format, dst->level_pitch[dst_level],
                 dst->level_pitch[dst_level] * dst->level_height[dst_level],
                 dstx, dsty, dstz, box->width, box->height, box->depth,
                 (uint8_t *)src_map + src->level_offset[src_level],
                 src->level_pitch[src_level],
                 src->level_pitch[src_level] * src->level_height[src_level],
                 box->x, box->y, box->z);

   grate_bo_unmap(src->bo);
   if (dst->bo != src->bo)
      grate_bo_unmap(dst->bo);
}

/* A flipped copy on the 2D engine. Gallium flips a blit with a negative width
 * or height in a box; the 2D engine's "fast rotate" mode (CONTROLSECOND
 * orientation, as in xf86-video-opentegra) flips one axis exactly. Measured on
 * the tablet (docs/grate-2d-flip.md): correct for 32 bits per pixel with pitches,
 * base offsets and the width and height multiples of 16 bytes (4 pixels), up to
 * 4096; anything else gives wrong pixels and writes past the destination.
 * Only one axis flipped, the same format on both sides, no scaling. Returns
 * false when the blit does not qualify. */
#define GRATE_2D_ORIENT_FLIP_X 0
#define GRATE_2D_ORIENT_FLIP_Y 1

static bool
grate_blit_flip_2d(struct grate_context *context,
                   const struct pipe_blit_info *info, struct grate_resource *src,
                   struct grate_resource *dst)
{
   const struct pipe_box *sb = &info->src.box, *db = &info->dst.box;
   bool flip_x = (sb->width < 0) != (db->width < 0);
   bool flip_y = (sb->height < 0) != (db->height < 0);

   if (flip_x == flip_y) /* both axes flipped: not measured */
      return false;
   if (src->tiled || dst->tiled || info->src.format != info->dst.format ||
       util_format_get_blocksize(info->src.format) != 4 ||
       info->mask != PIPE_MASK_RGBA || info->scissor_enable)
      return false;

   int w = abs(sb->width), h = abs(sb->height);
   if (w != abs(db->width) || h != abs(db->height) || w == 0 || h == 0 ||
       w > 4096 || h > 4096 || w % 4 || h % 4)
      return false;

   /* the rectangle of each side, flipped boxes start at their far edge */
   int sx = sb->width < 0 ? sb->x + sb->width : sb->x;
   int sy = sb->height < 0 ? sb->y + sb->height : sb->y;
   int dx = db->width < 0 ? db->x + db->width : db->x;
   int dy = db->height < 0 ? db->y + db->height : db->y;
   unsigned sw = u_minify(info->src.resource->width0, info->src.level);
   unsigned sh = u_minify(info->src.resource->height0, info->src.level);
   unsigned dw = u_minify(info->dst.resource->width0, info->dst.level);
   unsigned dh = u_minify(info->dst.resource->height0, info->dst.level);
   if (sx < 0 || sy < 0 || dx < 0 || dy < 0 || sx + w > (int)sw ||
       sy + h > (int)sh || dx + w > (int)dw || dy + h > (int)dh)
      return false;

   unsigned spitch = src->level_pitch[info->src.level];
   unsigned dpitch = dst->level_pitch[info->dst.level];
   unsigned soff = src->level_offset[info->src.level] + sy * spitch + sx * 4;
   unsigned doff = dst->level_offset[info->dst.level] + dy * dpitch + dx * 4;
   if (spitch % 16 || dpitch % 16 || soff % 16 || doff % 16)
      return false;

   struct grate_channel *gr2d = context->gr2d;
   uint32_t *ptr;
   if (grate_stream_begin(&gr2d->stream, &ptr) < 0)
      return false;

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x009, 0x9));
   GRATE_PUSHBUF_WORD(ptr, 0x00000037);            /* 0x009 - trigger */
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);            /* 0x00c - cmdsel */
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x01e, 0x5));
   /* 0x01e - controlsecond: [31:26] orientation, [25:24] 1 = source to
    * another destination */
   GRATE_PUSHBUF_WORD(ptr, (flip_y ? GRATE_2D_ORIENT_FLIP_Y
                                   : GRATE_2D_ORIENT_FLIP_X) << 26 |
                              1 << 24);
   GRATE_PUSHBUF_WORD(ptr, 0x000000cc);            /* 0x020 - ropfade */
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x046, 1));
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);            /* 0x046 - tilemode: linear */
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x02b, 0x108));
   GRATE_PUSHBUF_WORD(ptr, dpitch);                /* 0x02e - dstst */
   GRATE_PUSHBUF_WORD(ptr, spitch);                /* 0x033 - srcst */
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x02b, 1));
   grate_stream_push_reloc(&gr2d->stream, &ptr, dst->bo, doff);
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x031, 1));
   grate_stream_push_reloc(&gr2d->stream, &ptr, src->bo, soff);
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x01f, 1));
   /* 0x01f - controlmain: [29] fast rotate wait for read, [20] source colour
    * depth same, [17:16] destination 32 bits per pixel */
   GRATE_PUSHBUF_WORD(ptr, 1 << 29 | 1 << 20 | 2 << 16);
   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x037, 1));
   GRATE_PUSHBUF_WORD(ptr, (h - 1) << 16 | (w - 1)); /* 0x037 - srcsize, starts it */

   grate_stream_end(&gr2d->stream, &ptr);
   grate_stream_flush(&gr2d->stream, true);
   return true;
}

static void
grate_blit(struct pipe_context *pcontext, const struct pipe_blit_info *info)
{
   int err;
   uint32_t value;
   uint32_t *ptr;
   struct grate_context *context = grate_context(pcontext);
   struct grate_channel *gr2d = context->gr2d;
   struct grate_resource *dst, *src;

   dst = grate_resource(info->dst.resource);
   src = grate_resource(info->src.resource);

   if (getenv("GRATE_BLIT_LOG")) {
      /* every 2D blit, with the sizes of both levels, to find reads outside a
       * buffer (tegra-mc g2sr SMMU faults) */
      FILE *log = fopen(getenv("GRATE_BLIT_LOG"), "a");
      if (log) {
         const struct pipe_blit_info *i = info;
         unsigned sw = u_minify(i->src.resource->width0, i->src.level);
         unsigned sh = u_minify(i->src.resource->height0, i->src.level);
         unsigned dw = u_minify(i->dst.resource->width0, i->dst.level);
         unsigned dh = u_minify(i->dst.resource->height0, i->dst.level);
         fprintf(log,
                 "blit src %s lvl %u %ux%u box %d,%d %dx%d tiled %d | dst %s "
                 "lvl %u %ux%u box %d,%d %dx%d tiled %d | mask %x filter %u "
                 "scissor %d%s\n",
                 util_format_short_name(i->src.format), i->src.level, sw, sh,
                 i->src.box.x, i->src.box.y, i->src.box.width,
                 i->src.box.height, src->tiled,
                 util_format_short_name(i->dst.format), i->dst.level, dw, dh,
                 i->dst.box.x, i->dst.box.y, i->dst.box.width,
                 i->dst.box.height, dst->tiled, i->mask, i->filter,
                 i->scissor_enable,
                 (i->src.box.x < 0 || i->src.box.y < 0 ||
                  i->src.box.x + (int)i->dst.box.width > (int)sw ||
                  i->src.box.y + (int)i->dst.box.height > (int)sh ||
                  i->dst.box.x < 0 || i->dst.box.y < 0 ||
                  i->dst.box.x + i->dst.box.width > dw ||
                  i->dst.box.y + i->dst.box.height > dh ||
                  i->src.box.width != i->dst.box.width ||
                  i->src.box.height != i->dst.box.height)
                    ? " SUSPECT"
                    : "");
         fclose(log);
      }
   }

   if (info->src.box.width < 0 || info->src.box.height < 0 ||
       info->dst.box.width < 0 || info->dst.box.height < 0) {
      if (!grate_blit_flip_2d(context, info, src, dst))
         grate_unimplemented(); /* TODO: flip on the 3D engine or the CPU */
      return;
   }

   /* The plain 2D copy reads src.box at the size of dst.box: a box outside
    * the source, or another size, reads memory that is not the source (the
    * tegra-mc g2sr SMMU faults), so refuse it. */
   {
      unsigned sw = u_minify(info->src.resource->width0, info->src.level);
      unsigned sh = u_minify(info->src.resource->height0, info->src.level);
      unsigned dw = u_minify(info->dst.resource->width0, info->dst.level);
      unsigned dh = u_minify(info->dst.resource->height0, info->dst.level);
      if (info->src.box.x < 0 || info->src.box.y < 0 || info->dst.box.x < 0 ||
          info->dst.box.y < 0 ||
          info->src.box.x + info->dst.box.width > sw ||
          info->src.box.y + info->dst.box.height > sh ||
          info->dst.box.x + info->dst.box.width > dw ||
          info->dst.box.y + info->dst.box.height > dh ||
          info->src.box.width != info->dst.box.width ||
          info->src.box.height != info->dst.box.height) {
         grate_unimplemented();
         return;
      }
   }

   err = grate_stream_begin(&gr2d->stream, &ptr);
   if (err < 0) {
      fprintf(stderr, "grate_stream_begin() failed: %d\n", err);
      return;
   }

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x009, 0x9));
   GRATE_PUSHBUF_WORD(ptr, 0x0000003a);            /* 0x009 - trigger */
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);            /* 0x00c - cmdsel */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x01e, 0x7));
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);            /* 0x01e - controlsecond */
   /*
    * [20:20] source color depth (0: mono, 1: same)
    * [17:16] destination color depth (0: 8 bpp, 1: 16 bpp, 2: 32 bpp)
    */

   value = 1 << 20;
   switch (util_format_get_blocksize(dst->b.format)) {
   case 1:
      value |= 0 << 16;
      break;
   case 2:
      value |= 1 << 16;
      break;
   case 4:
      value |= 2 << 16;
      break;
   default:
      assert(0);
   }

   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x01f - controlmain */
   GRATE_PUSHBUF_WORD(ptr, 0x000000cc);            /* 0x020 - ropfade */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x046, 1));

   /*
    * [20:20] destination write tile mode (0: linear, 1: tiled)
    * [ 0: 0] tile mode Y/RGB (0: linear, 1: tiled)
    */
   value = (dst->tiled << 20) | src->tiled;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x046 - tilemode */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x02b, 0xe149));
   grate_stream_push_reloc(&gr2d->stream, &ptr, dst->bo,
                           dst->level_offset[info->dst.level]);

   GRATE_PUSHBUF_WORD(ptr, dst->level_pitch[info->dst.level]);

   grate_stream_push_reloc(&gr2d->stream, &ptr, src->bo,
                           src->level_offset[info->src.level]);

   GRATE_PUSHBUF_WORD(ptr, src->level_pitch[info->src.level]);

   value = info->dst.box.height << 16 | info->dst.box.width;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x038 - dstsize */

   value = info->src.box.y << 16 | info->src.box.x;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x039 - srcps */

   value = info->dst.box.y << 16 | info->dst.box.x;
   GRATE_PUSHBUF_WORD(ptr, value);                 /* 0x03a - dstps */

   grate_stream_end(&gr2d->stream, &ptr);

   grate_stream_flush(&gr2d->stream, true);
}

static uint32_t
pack_color(enum pipe_format format, const float *rgba)
{
   union util_color uc;
   util_pack_color(rgba, format, &uc);
   return uc.ui[0];
}

static void
fill(struct grate_stream *stream, uint32_t **ptrp, struct grate_resource *dst,
     unsigned base_offset, unsigned pitch, uint32_t fill_value, int blocksize,
     unsigned dstx, unsigned dsty, unsigned width, unsigned height)
{
   uint32_t value;
   uint32_t *ptr = *ptrp;

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x09, 0x09));
   GRATE_PUSHBUF_WORD(ptr, 0x0000003a);           /* 0x009 - trigger */
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);           /* 0x00C - cmdsel */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x1e, 0x07));
   GRATE_PUSHBUF_WORD(ptr, 0x00000000);           /* 0x01e - controlsecond */

   value  = 1 << 6; /* fill mode */
   value |= 1 << 2; /* turbofill */
   switch (blocksize) {
   case 1:
      value |= 0 << 16;
      break;
   case 2:
      value |= 1 << 16;
      break;
   case 4:
      value |= 2 << 16;
      break;
   default:
      UNREACHABLE("invalid blocksize");
   }
   GRATE_PUSHBUF_WORD(ptr, value);           /* 0x01f - controlmain */

   GRATE_PUSHBUF_WORD(ptr, 0x000000cc);      /* 0x020 - ropfade */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x2b, 0x09));
   grate_stream_push_reloc(stream, &ptr, dst->bo,
                           base_offset); /* 0x02b - dstba */
   GRATE_PUSHBUF_WORD(ptr, pitch);       /* 0x02e - dstst */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x35, 1));

   GRATE_PUSHBUF_WORD(ptr, fill_value);           /* 0x035 - srcfgc */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_nonincr(0x46, 1));
   GRATE_PUSHBUF_WORD(ptr, dst->tiled << 20);     /* 0x046 - tilemode */

   GRATE_PUSHBUF_WORD(ptr, host1x_opcode_mask(0x38, 0x05));
   GRATE_PUSHBUF_WORD(ptr, height << 16 | width); /* 0x038 - dstsize */
   GRATE_PUSHBUF_WORD(ptr, dsty << 16 | dstx);    /* 0x03a - dstps */
   
   *ptrp = ptr;
}

/* Clears a depth/stencil surface. The emulated Z24 formats are two planes
 * (Z16NL depth, S8 stencil), each filled on its own, so clearing only one of
 * them leaves the other alone; every other format is one fill of the packed
 * value. clear_flags is PIPE_CLEAR_DEPTH and/or PIPE_CLEAR_STENCIL.
 */
static void
clear_zs(struct grate_stream *stream, uint32_t **ptrp,
         struct grate_resource *res, enum pipe_format format,
         unsigned clear_flags, double depth, unsigned stencil, unsigned x,
         unsigned y, unsigned width, unsigned height)
{
   if (!grate_format_is_emulated_z24(format)) {
      fill(stream, ptrp, res, 0, res->pitch,
           util_pack_z_stencil(format, depth, stencil),
           util_format_get_blocksize(format), x, y, width, height);
      return;
   }

   if (clear_flags & PIPE_CLEAR_DEPTH)
      fill(stream, ptrp, res, 0, res->pitch, grate_pack_z16nl(depth), 2, x, y,
           width, height);

   if ((clear_flags & PIPE_CLEAR_STENCIL) && res->stencil_offset)
      fill(stream, ptrp, res, res->stencil_offset, res->stencil_pitch,
           stencil & 0xff, 1, x, y, width, height);
}

static void
grate_clear(struct pipe_context *pcontext, unsigned int buffers,
            uint32_t color_clear_mask, uint8_t stencil_clear_mask,
            const struct pipe_scissor_state *scissor_state,
            const union pipe_color_union *color, double depth,
            unsigned int stencil)
{
   (void) color_clear_mask;
   (void) stencil_clear_mask;
   struct grate_context *context = grate_context(pcontext);
   struct grate_stream *stream = &context->gr2d->stream;
   uint32_t *ptr;
   int err;
   struct pipe_framebuffer_state *fb;

   fb = &context->framebuffer.base;

   err = grate_stream_begin(stream, &ptr);
   if (err < 0) {
      grate_loge("grate_stream_begin() failed: %d\n", err);
      return;
   }

   if (buffers & PIPE_CLEAR_COLOR) {
      int i;
      for (i = 0; i < fb->nr_cbufs; ++i) {
         struct pipe_surface *dst = &fb->cbufs[i];
         fill(stream, &ptr, grate_resource(dst->texture),
              grate_resource(dst->texture)->level_offset[dst->level],
              grate_resource(dst->texture)->level_pitch[dst->level],
              pack_color(dst->format, color->f),
              util_format_get_blocksize(dst->format), 0, 0, fb->width,
              fb->height);
      }
   }

   if (buffers & PIPE_CLEAR_DEPTH || buffers & PIPE_CLEAR_STENCIL) {
      if (fb->zsbuf.texture) {
         /* TODO: handle the case where both are not set! (done for the Z24
          * formats, which have separate planes) */
         clear_zs(stream, &ptr, grate_resource(fb->zsbuf.texture),
                  fb->zsbuf.format, buffers & PIPE_CLEAR_DEPTHSTENCIL, depth,
                  stencil, 0, 0, fb->width, fb->height);
      }
   }
   
   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, true);
}

static void
grate_clear_render_target(struct pipe_context *pipe,
                          struct pipe_surface *dst,
                          const union pipe_color_union *color,
                          unsigned dstx, unsigned dsty,
                          unsigned width, unsigned height,
                          bool render_condition_enabled)
{
   struct grate_context *context = grate_context(pipe);
   struct grate_stream *stream = &context->gr2d->stream;
   uint32_t *ptr;
   int err;
   assert(!render_condition_enabled);
   
   err = grate_stream_begin(stream, &ptr);
   if (err < 0) {
      grate_loge("grate_stream_begin() failed: %d\n", err);
      return;
   }

   fill(&context->gr2d->stream, &ptr, grate_resource(dst->texture),
        grate_resource(dst->texture)->level_offset[dst->level],
        grate_resource(dst->texture)->level_pitch[dst->level],
        pack_color(dst->format, color->f),
        util_format_get_blocksize(dst->format), dstx, dsty, width, height);

   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, true);
}

static void
grate_clear_depth_stencil(struct pipe_context *pipe,
                          struct pipe_surface *dst,
                          unsigned clear_flags,
                          double depth,
                          unsigned stencil,
                          unsigned dstx, unsigned dsty,
                          unsigned width, unsigned height,
                          bool render_condition_enabled)
{
   struct grate_context *context = grate_context(pipe);
   struct grate_stream *stream = &context->gr2d->stream;
   uint32_t *ptr;
   int err;
   assert(!render_condition_enabled);
   
   err = grate_stream_begin(stream, &ptr);
   if (err < 0) {
      grate_loge("grate_stream_begin() failed: %d\n", err);
      return;
   }

   clear_zs(&context->gr2d->stream, &ptr, grate_resource(dst->texture),
            dst->format, clear_flags, depth, stencil, dstx, dsty, width,
            height);

   grate_stream_end(stream, &ptr);
   grate_stream_flush(stream, true);
}

static void
grate_flush_resource(struct pipe_context *ctx, struct pipe_resource *resource)
{
   int err = grate_bo_wait(grate_resource(resource)->bo);
   if (err)
      grate_loge("resource wait failed: %d\n", err);
}

void
grate_context_resource_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->buffer_map = grate_resource_transfer_map;
   pcontext->texture_map = grate_resource_transfer_map;
   pcontext->transfer_flush_region = grate_resource_transfer_flush_region;
   pcontext->buffer_unmap = grate_resource_transfer_unmap;
   pcontext->texture_unmap = grate_resource_transfer_unmap;
   pcontext->buffer_subdata = u_default_buffer_subdata;
   pcontext->texture_subdata = u_default_texture_subdata;

   pcontext->resource_copy_region = grate_resource_copy_region;
   pcontext->blit = grate_blit;
   pcontext->clear = grate_clear;
   pcontext->clear_buffer = u_default_clear_buffer;
   pcontext->flush_resource = grate_flush_resource;
   pcontext->resource_release = u_default_resource_release;
   pcontext->clear_render_target = grate_clear_render_target;
   pcontext->clear_depth_stencil = grate_clear_depth_stencil;
}
