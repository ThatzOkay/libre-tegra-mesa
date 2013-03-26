#include <stdio.h>

#include "drm-uapi/drm_fourcc.h"

#include "util/u_memory.h"
#include "util/u_screen.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_resource.h"
#include "grate_screen.h"

#include "opentegra_lib.h"

static const struct debug_named_value debug_options[] = {
   { "unimplemented", GRATE_DEBUG_UNIMPLEMENTED,
     "Print unimplemented functions" },
   { NULL }
};

DEBUG_GET_ONCE_FLAGS_OPTION(grate_debug, "GRATE_DEBUG", debug_options, 0)
uint32_t grate_debug;

static void
grate_screen_destroy(struct pipe_screen *pscreen)
{
   struct grate_screen *screen = grate_screen(pscreen);

   drm_tegra_close(screen->drm);
   FREE(screen);
}

static const char *
grate_screen_get_name(struct pipe_screen *pscreen)
{
   return "Tegra";
}

static const char *
grate_screen_get_vendor(struct pipe_screen *pscreen)
{
   return "Grate";
}

static const char *
grate_screen_get_device_vendor(struct pipe_screen *pscreen)
{
   return "NVIDIA";
}

static int
grate_screen_get_screen_fd(struct pipe_screen *pscreen)
{
   return grate_screen(pscreen)->fd;
}

static int
grate_screen_get_param(struct pipe_screen *pscreen, enum pipe_cap param)
{
   switch (param) {
   case PIPE_CAP_NPOT_TEXTURES:
      return 1; /* not really, but mesa requires it for now! */

   case PIPE_CAP_MAX_RENDER_TARGETS:
      return 8; /* ??? */

   case PIPE_CAP_MAX_TEXTURE_2D_SIZE:
      return 2048;

   case PIPE_CAP_MAX_TEXTURE_3D_LEVELS:
      return 0;

   case PIPE_CAP_MAX_TEXTURE_CUBE_LEVELS:
      return 16; /* ??? */

   case PIPE_CAP_SUPPORTED_PRIM_MODES_WITH_RESTART:
      return 0;

   case PIPE_CAP_SUPPORTED_PRIM_MODES:
      return BITFIELD_BIT(MESA_PRIM_POINTS) |
             BITFIELD_BIT(MESA_PRIM_LINES) |
             BITFIELD_BIT(MESA_PRIM_LINE_LOOP) |
             BITFIELD_BIT(MESA_PRIM_LINE_STRIP) |
             BITFIELD_BIT(MESA_PRIM_TRIANGLES) |
             BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP) |
             BITFIELD_BIT(MESA_PRIM_TRIANGLE_FAN);

   case PIPE_CAP_BLEND_EQUATION_SEPARATE:
      return 1;

   case PIPE_CAP_FRAGMENT_SHADER_TEXTURE_LOD:
   case PIPE_CAP_FRAGMENT_SHADER_DERIVATIVES:
      return 1; /* well, not quite. but perhaps close enough? */

   case PIPE_CAP_MIN_TEXEL_OFFSET:
   case PIPE_CAP_MAX_TEXEL_OFFSET:
      return 0;

   case PIPE_CAP_VERTEX_COLOR_UNCLAMPED:
      return 1; /* probably irrelevant for GLES2 */

   case PIPE_CAP_VERTEX_COLOR_CLAMPED:
      return 0; /* probably irrelevant for GLES2 */

   case PIPE_CAP_GLSL_FEATURE_LEVEL:
      return 120; /* no clue */

   case PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT:
      return 4; /* DWORD aligned, can do pure data GATHER */

   case PIPE_CAP_TEXTURE_TRANSFER_MODES:
      return PIPE_TEXTURE_TRANSFER_BLIT;

   case PIPE_CAP_MIXED_FRAMEBUFFER_SIZES:
      return 1;

   case PIPE_CAP_BUFFER_MAP_PERSISTENT_COHERENT: /* dunno */
      return 0;

   case PIPE_CAP_VENDOR_ID:
      return 0x10de;

   case PIPE_CAP_DEVICE_ID:
      return 0xFFFFFFFF;

   case PIPE_CAP_ACCELERATED:
      return 1;

   case PIPE_CAP_VIDEO_MEMORY:
      return 0;

   case PIPE_CAP_UMA:
      return 1;

   case PIPE_CAP_MAX_VERTEX_ATTRIB_STRIDE:
      return (1 << 24) - 1;

   case PIPE_CAP_MIXED_COLOR_DEPTH_BITS:
      return 1; /* probably true ? */

   case PIPE_CAP_FBFETCH: /* TODO: supported, but let's enable later */
   case PIPE_CAP_CAN_BIND_CONST_BUFFER_AS_VERTEX: /* TODO: probably */
   case PIPE_CAP_ALLOW_MAPPED_BUFFERS_DURING_EXECUTION: /* TODO: probably */
      return 0;

   case PIPE_CAP_MAX_VARYINGS:
      return 16;

   default:
      return u_pipe_screen_get_param_defaults(pscreen, param);
   }
}

static float
grate_screen_get_paramf(struct pipe_screen *pscreen,
                        enum pipe_capf param)
{
   switch (param) {
   case PIPE_CAPF_MIN_LINE_WIDTH:
   case PIPE_CAPF_MIN_LINE_WIDTH_AA:
   case PIPE_CAPF_MIN_POINT_SIZE:
   case PIPE_CAPF_MIN_POINT_SIZE_AA:
      return 1.0f; /* no clue */

   case PIPE_CAPF_MAX_LINE_WIDTH:
   case PIPE_CAPF_MAX_LINE_WIDTH_AA:
   case PIPE_CAPF_MAX_POINT_SIZE:
   case PIPE_CAPF_MAX_POINT_SIZE_AA:
      return 8192.0f; /* no clue */

   case PIPE_CAPF_LINE_WIDTH_GRANULARITY:
   case PIPE_CAPF_POINT_SIZE_GRANULARITY:
      return 1.0 / 16; /* not a real limit, HW uses floats... but helps caching CSOs */

   case PIPE_CAPF_MAX_TEXTURE_ANISOTROPY:
      return 0.0f;

   case PIPE_CAPF_MAX_TEXTURE_LOD_BIAS:
      return 16.0f;

   case PIPE_CAPF_MIN_CONSERVATIVE_RASTER_DILATE:
   case PIPE_CAPF_MAX_CONSERVATIVE_RASTER_DILATE:
   case PIPE_CAPF_CONSERVATIVE_RASTER_DILATE_GRANULARITY:
      return 0.0f;

   default:
      fprintf(stdout, "%s: unsupported parameter: %d\n", __func__, param);
      return 0.0f;
   }
}

static int
grate_screen_get_shader_param(struct pipe_screen *pscreen,
                              enum pipe_shader_type shader,
                              enum pipe_shader_cap param)
{
   switch (shader) {
   case PIPE_SHADER_VERTEX:
      switch (param) {

      case PIPE_SHADER_CAP_MAX_INSTRUCTIONS:
      case PIPE_SHADER_CAP_MAX_ALU_INSTRUCTIONS:
         return 1024;

      /* no vertex-texturing */
      case PIPE_SHADER_CAP_MAX_TEX_INSTRUCTIONS:
      case PIPE_SHADER_CAP_MAX_TEX_INDIRECTIONS:
      case PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS:
      case PIPE_SHADER_CAP_MAX_SAMPLER_VIEWS:
         return 0;

      case PIPE_SHADER_CAP_MAX_CONTROL_FLOW_DEPTH:
      case PIPE_SHADER_CAP_CONT_SUPPORTED:
      case PIPE_SHADER_CAP_SUBROUTINES:
         return 0;

      case PIPE_SHADER_CAP_MAX_INPUTS:
      case PIPE_SHADER_CAP_MAX_OUTPUTS:
         return 16;

      case PIPE_SHADER_CAP_MAX_CONST_BUFFER0_SIZE:
      case PIPE_SHADER_CAP_MAX_CONST_BUFFERS:
         return 1024;

      case PIPE_SHADER_CAP_MAX_TEMPS:
         return 64 * 4; /* 64 vec4s */

      /* cannot index attributes, varyings nor GPRs */
      case PIPE_SHADER_CAP_INDIRECT_INPUT_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_OUTPUT_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_TEMP_ADDR:
         return 0;

      case PIPE_SHADER_CAP_INDIRECT_CONST_ADDR:
         return 1; /* can index constant registers */

      case PIPE_SHADER_CAP_INTEGERS:
      case PIPE_SHADER_CAP_MAX_SHADER_IMAGES:
         return 0;

      case PIPE_SHADER_CAP_TGSI_SQRT_SUPPORTED:
         return 1;

      case PIPE_SHADER_CAP_MAX_SHADER_BUFFERS:
         return 0;

      case PIPE_SHADER_CAP_TGSI_ANY_INOUT_DECL_RANGE:
         return 0;

      case PIPE_SHADER_CAP_SUPPORTED_IRS:
         return (1 << PIPE_SHADER_IR_TGSI) | (1 << PIPE_SHADER_IR_NIR);

      case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTERS:
      case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTER_BUFFERS:
          return 0;

      case PIPE_SHADER_CAP_FP16:
      case PIPE_SHADER_CAP_FP16_DERIVATIVES:
      case PIPE_SHADER_CAP_FP16_CONST_BUFFERS:
      case PIPE_SHADER_CAP_INT16:
      case PIPE_SHADER_CAP_GLSL_16BIT_CONSTS:
          return 0;

      default:
         fprintf(stdout, "%s: unsupported vertex-shader parameter: %d\n", __func__, param);
         return 0;
      }

   case PIPE_SHADER_FRAGMENT:

      switch (param) {
      case PIPE_SHADER_CAP_MAX_INSTRUCTIONS:
         return 4 * 128;

      case PIPE_SHADER_CAP_MAX_ALU_INSTRUCTIONS:
         return 4 * 128;

      case PIPE_SHADER_CAP_MAX_TEX_INSTRUCTIONS:
         return 128;

      case PIPE_SHADER_CAP_MAX_TEX_INDIRECTIONS:
         return 128;

      /* no control flow */
      case PIPE_SHADER_CAP_MAX_CONTROL_FLOW_DEPTH:
      case PIPE_SHADER_CAP_CONT_SUPPORTED:
      case PIPE_SHADER_CAP_SUBROUTINES:
         return 0;

      case PIPE_SHADER_CAP_MAX_INPUTS:
      case PIPE_SHADER_CAP_MAX_OUTPUTS:
         return 16;

      case PIPE_SHADER_CAP_MAX_CONST_BUFFER0_SIZE:
      case PIPE_SHADER_CAP_MAX_CONST_BUFFERS:
         return 32;

      case PIPE_SHADER_CAP_MAX_TEMPS:
         return 16; /* scalars */

      /* no indirection */
      case PIPE_SHADER_CAP_INDIRECT_INPUT_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_OUTPUT_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_TEMP_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_CONST_ADDR:
         return 0;

      case PIPE_SHADER_CAP_INTEGERS:
      case PIPE_SHADER_CAP_MAX_SHADER_IMAGES:
         return 0;

      case PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS:
      case PIPE_SHADER_CAP_MAX_SAMPLER_VIEWS:
         return 16;

      case PIPE_SHADER_CAP_TGSI_SQRT_SUPPORTED:
         return 1;

      case PIPE_SHADER_CAP_TGSI_ANY_INOUT_DECL_RANGE:
         return 0;

      case PIPE_SHADER_CAP_MAX_SHADER_BUFFERS:
         return 0;

      case PIPE_SHADER_CAP_SUPPORTED_IRS:
         return (1 << PIPE_SHADER_IR_TGSI) | (1 << PIPE_SHADER_IR_NIR);

      case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTERS:
      case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTER_BUFFERS:
          return 0;

      case PIPE_SHADER_CAP_FP16:
      case PIPE_SHADER_CAP_FP16_DERIVATIVES:
      case PIPE_SHADER_CAP_FP16_CONST_BUFFERS:
      case PIPE_SHADER_CAP_INT16:
      case PIPE_SHADER_CAP_GLSL_16BIT_CONSTS:
          return 0;

      default:
         fprintf(stdout, "%s: unsupported fragment-shader parameter: %d\n", __func__, param);
         return 0;
      }
      break;

   case PIPE_SHADER_GEOMETRY:
   case PIPE_SHADER_TESS_CTRL:
   case PIPE_SHADER_TESS_EVAL:
   case PIPE_SHADER_COMPUTE:
      return 0;

   default:
      fprintf(stdout, "%s: unknown shader type: %u\n", __func__, shader);
      return 0;
   }
}


static bool
grate_screen_is_format_supported(struct pipe_screen *pscreen,
                                 enum pipe_format format,
                                 enum pipe_texture_target target,
                                 unsigned sample_count,
                                 unsigned storage_sample_count,
                                 unsigned usage)
{
   if (usage & PIPE_BIND_RENDER_TARGET) {
      switch (format) {
      case PIPE_FORMAT_B8G8R8A8_UNORM:
      case PIPE_FORMAT_B8G8R8X8_UNORM:
      case PIPE_FORMAT_A8R8G8B8_UNORM:
      case PIPE_FORMAT_X8R8G8B8_UNORM:
         break;
      default:
         return false;
      }
   }

   if (usage & PIPE_BIND_DEPTH_STENCIL) {
      switch (format) {
      case PIPE_FORMAT_Z16_UNORM:
      case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      case PIPE_FORMAT_S8_UINT:
         break;
      default:
         return false;
      }
   }

   return true;
}

static void
grate_screen_fence_reference(struct pipe_screen *pscreen,
                             struct pipe_fence_handle **ptr,
                             struct pipe_fence_handle *fence)
{
   unimplemented();
}

static bool
grate_screen_fence_finish(struct pipe_screen *screen,
                          struct pipe_context *ctx,
                          struct pipe_fence_handle *fence,
                          uint64_t timeout)
{
   unimplemented();
   return false;
}

static const uint64_t grate_available_modifiers[] = {
   DRM_FORMAT_MOD_LINEAR,
};

static void
grate_screen_query_dmabuf_modifiers(struct pipe_screen *pscreen,
                                    enum pipe_format format, int max,
                                    uint64_t *modifiers,
                                    unsigned int *external_only,
                                    int *count)
{
   int num_modifiers = ARRAY_SIZE(grate_available_modifiers);

   if (!modifiers) {
      *count = num_modifiers;
      return;
   }

   *count = MIN2(max, num_modifiers);
   for (int i = 0; i < *count; i++) {
      modifiers[i] = grate_available_modifiers[i];
      if (external_only)
         external_only[i] = false;
   }
}

static bool
grate_screen_is_dmabuf_modifier_supported(struct pipe_screen *pscreen,
                                          uint64_t modifier,
                                          enum pipe_format format,
                                          bool *external_only)
{
   for (int i = 0; i < ARRAY_SIZE(grate_available_modifiers); i++) {
      if (grate_available_modifiers[i] == modifier) {
         if (external_only)
            *external_only = false;

         return true;
      }
   }

   return false;
}

struct pipe_screen *
grate_screen_create(int fd)
{
   struct drm_tegra_channel *drm_channel;
   struct grate_screen *screen;
   int err;

   screen = CALLOC_STRUCT(grate_screen);
   if (!screen)
      return NULL;

   screen->fd = fd;
   err = drm_tegra_new(&screen->drm, fd);
   if (err) {
      FREE(screen);
      return NULL;
   }

   err = drm_tegra_channel_open(&drm_channel, screen->drm, DRM_TEGRA_GR3D);
   if (err) {
      drm_tegra_close(screen->drm);
      FREE(screen);
      return NULL;
   }

   drm_tegra_channel_close(drm_channel);

   grate_debug = debug_get_option_grate_debug();

   screen->base.destroy = grate_screen_destroy;
   screen->base.get_name = grate_screen_get_name;
   screen->base.get_vendor = grate_screen_get_vendor;
   screen->base.get_device_vendor = grate_screen_get_device_vendor;
   screen->base.get_screen_fd = grate_screen_get_screen_fd;
   screen->base.get_param = grate_screen_get_param;
   screen->base.get_paramf = grate_screen_get_paramf;
   screen->base.get_shader_param = grate_screen_get_shader_param;
   screen->base.context_create = grate_screen_context_create;
   screen->base.is_format_supported = grate_screen_is_format_supported;
   screen->base.query_dmabuf_modifiers = grate_screen_query_dmabuf_modifiers;
   screen->base.is_dmabuf_modifier_supported = grate_screen_is_dmabuf_modifier_supported;

   /* fence functions */
   screen->base.fence_reference = grate_screen_fence_reference;
   screen->base.fence_finish = grate_screen_fence_finish;

   grate_screen_resource_init(&screen->base);

   return &screen->base;
}
