#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util/format/u_format.h"
#include "util/u_bitcast.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_framebuffer.h"

#include "grate_compiler.h"
#include "grate_common.h"
#include "grate_context.h"
#include "grate_program.h"
#include "grate_resource.h"
#include "grate_state.h"

#include "host1x01_hardware.h"

static void
grate_set_sample_mask(struct pipe_context *pcontext,
                      unsigned int sample_mask)
{
   grate_unimplemented();
}

static void
grate_set_constant_buffer(struct pipe_context *pcontext, enum mesa_shader_stage shader,
                          uint index, const struct pipe_constant_buffer *buffer)
{
   struct grate_context *context = grate_context(pcontext);

   assert(index == 0);
   assert(!buffer || buffer->user_buffer);

   util_copy_constant_buffer(&context->constant_buffer[shader], buffer);
}

static void grate_add_render_target(struct grate_context *context,
                                    const struct pipe_surface *ref,
                                    unsigned int slot)
{
   if (!ref->texture) {
      fprintf(stderr, "%s: texture at %p null!", __func__, ref);
      assert(0);
      return;
   }
   if (slot >= REG_TGR3D_GLOBAL_SURFDESC_LENGTH) {
      fprintf(stderr, "%s: Reached max render targets!", __func__);
      assert(0);
      return;
   }
   struct grate_resource *res = grate_resource(ref->texture);
   uint32_t rt_params;
   
   rt_params  = TGR3D_GLOBAL_SURFDESC_SURF_FORMAT(res->format);
   rt_params |= TGR3D_GLOBAL_SURFDESC_ARRAY_STRIDE(res->pitch);
   rt_params |= TGR3D_GLOBAL_SURFDESC_STRUCTURE(res->tiled);
   
   context->framebuffer.rt_params[slot] = rt_params;
   context->framebuffer.rt_bos[slot] = res->bo;
   context->framebuffer.rt_mask |= 1u << slot;
   context->framebuffer.num_rts = MAX2(context->framebuffer.num_rts, slot + 1);
}

static void
grate_set_framebuffer_state(struct pipe_context *pcontext,
                            const struct pipe_framebuffer_state *framebuffer)
{
   struct grate_context *context = grate_context(pcontext);
   struct pipe_framebuffer_state *cso = &context->framebuffer.base;
   context->framebuffer.rt_mask = 0;
   context->framebuffer.num_rts = 0;

   util_copy_framebuffer_state(cso, framebuffer);

   /* Slot 0 is the fixed depth target. Keep colour output i at i + 1,
    * even without depth, matching the fragment compiler's DW stores.
    */
   for (unsigned int i = 0; i < framebuffer->nr_cbufs; i++) {
      if (framebuffer->cbufs[i].texture)
         grate_add_render_target(context, &framebuffer->cbufs[i], i + 1);
   }
   
   if (framebuffer->zsbuf.texture) {
      grate_add_render_target(context, &framebuffer->zsbuf, 0);
   }

   /* prepare the scissor-registers for the non-scissor case */
   context->no_scissor[0]  = host1x_opcode_incr(REG_TGR3D_SU_SCISSOR_X, 2);
   context->no_scissor[1]  = TGR3D_SU_SCISSOR_X_MIN(0);
   context->no_scissor[1] |= TGR3D_SU_SCISSOR_X_MAX(framebuffer->width);
   context->no_scissor[2]  = TGR3D_SU_SCISSOR_Y_MIN(0);
   context->no_scissor[2] |= TGR3D_SU_SCISSOR_Y_MAX(framebuffer->height);
}

static void
grate_set_polygon_stipple(struct pipe_context *pcontext,
                          const struct pipe_poly_stipple *stipple)
{
   grate_unimplemented();
}

static void
grate_set_scissor_states(struct pipe_context *pcontext,
                         unsigned start_slot,
                         unsigned num_scissors,
                         const struct pipe_scissor_state * scissors)
{
   assert(num_scissors == 1);
   grate_unimplemented();
}

static void
grate_set_viewport_states(struct pipe_context *pcontext,
                          unsigned start_slot,
                          unsigned num_viewports,
                          const struct pipe_viewport_state *viewports)
{
   struct grate_context *context = grate_context(pcontext);
   /* 2.0f ** -21, exact in float -- powf() isn't a compile-time constant
    * expression in standard C; GCC accepts it as an extension, clang
    * doesn't. */
   static const float zeps = 0x1p-21f;
   unsigned int hw_scale;

   if (context->drm->soc_id == DRM_TEGRA_SOC_T114)
      hw_scale = 0xFFFFFF;
   else
      hw_scale = 0xFFFFF;

   assert(num_viewports == 1);
   assert(start_slot == 0);

   context->viewport[0] = host1x_opcode_incr(REG_TGR3D_SU_VIEWPORT_X, 6);
   context->viewport[1] = u_bitcast_f2u(viewports[0].translate[0] * 16.0f);
   context->viewport[2] = u_bitcast_f2u(viewports[0].translate[1] * 16.0f);
   context->viewport[3] = u_bitcast_f2u(viewports[0].translate[2] - zeps);
   context->viewport[4] = u_bitcast_f2u(viewports[0].scale[0] * 16.0f);
   context->viewport[5] = u_bitcast_f2u(viewports[0].scale[1] * 16.0f);
   context->viewport[6] = u_bitcast_f2u(viewports[0].scale[2] - zeps);

   uint32_t depth_near = (viewports[0].translate[2] - viewports[0].scale[2]) * hw_scale;
   uint32_t depth_far = (viewports[0].translate[2] + viewports[0].scale[2]) * hw_scale;
   context->viewport[7] = host1x_opcode_incr(REG_TGR3D_QR_Z_MIN, 2);
   context->viewport[8] = depth_near;
   context->viewport[9] = depth_far;

   assert(viewports[0].scale[0] >= 0.0f);
   float max_x = fabs(viewports[0].translate[0]);
   float max_y = fabs(viewports[0].translate[1]);
   float scale_x = viewports[0].scale[0];
   float scale_y = fabs(viewports[0].scale[1]);
   context->guardband[0] = host1x_opcode_incr(REG_TGR3D_SU_GUARDBAND_W, 3);
   context->guardband[1] = u_bitcast_f2u((3967 - max_x) / scale_x);
   context->guardband[2] = u_bitcast_f2u((3967 - max_y) / scale_y);
   context->guardband[3] = u_bitcast_f2u(6.99);

   context->y_invert = viewports[0].scale[1] < 0.0f;
}

static void
grate_set_vertex_buffers(struct pipe_context *pcontext,
                         unsigned count,
                         const struct pipe_vertex_buffer *buffer)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_vertexbuf_state *vbs = &context->vbs;

   util_set_vertex_buffers_mask(vbs->vb, &vbs->enabled, buffer, count);
   vbs->count = util_last_bit(vbs->enabled);
}


static void
grate_set_sampler_views(struct pipe_context *pctx, mesa_shader_stage shader,
                        unsigned start_slot, unsigned num_views,
                        unsigned unbind_num_trailing_slots,
                        struct pipe_sampler_view **views)
{
   struct grate_context *ctx = grate_context(pctx);
   assert(shader == MESA_SHADER_FRAGMENT);
   assert(start_slot + num_views + unbind_num_trailing_slots <= 16);
   for (unsigned i = 0; i < num_views; i++)
      pipe_sampler_view_reference(&ctx->sampler_views[start_slot + i], views ? views[i] : NULL);
   for (unsigned i = 0; i < unbind_num_trailing_slots; i++)
      pipe_sampler_view_reference(&ctx->sampler_views[start_slot + num_views + i], NULL);
}

static void
grate_set_blend_color(struct pipe_context *pctx,
                      const struct pipe_blend_color *blend_color)
{
   grate_unimplemented();
}

static void
grate_set_stencil_ref(struct pipe_context *pctx,
                      const struct pipe_stencil_ref stencil_ref)
{
   grate_unimplemented();
}

static void
grate_fp_state_bind(struct pipe_context *pctx, void *hwcso)
{
   grate_unimplemented();
}

static void
grate_vp_state_bind(struct pipe_context *pctx, void *hwcso)
{
   grate_unimplemented();
}

void
grate_context_state_init(struct pipe_context *pcontext)
{
   pcontext->set_blend_color = grate_set_blend_color;
   pcontext->set_stencil_ref = grate_set_stencil_ref;
   pcontext->bind_fs_state = grate_fp_state_bind;
   pcontext->bind_vs_state = grate_vp_state_bind;
   pcontext->set_sample_mask = grate_set_sample_mask;
   pcontext->set_constant_buffer = grate_set_constant_buffer;
   pcontext->set_framebuffer_state = grate_set_framebuffer_state;
   pcontext->set_polygon_stipple = grate_set_polygon_stipple;
   pcontext->set_scissor_states = grate_set_scissor_states;
   pcontext->set_viewport_states = grate_set_viewport_states;
   pcontext->set_sampler_views = grate_set_sampler_views;
   pcontext->set_vertex_buffers = grate_set_vertex_buffers;
}

static void *
grate_create_blend_state(struct pipe_context *pcontext,
                         const struct pipe_blend_state *template)
{
   struct pipe_blend_state *so = CALLOC_STRUCT(pipe_blend_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_blend_state(struct pipe_context *pcontext, void *so)
{
   grate_unimplemented();
}

static void
grate_delete_blend_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_blend_init(struct pipe_context *pcontext)
{
   pcontext->create_blend_state = grate_create_blend_state;
   pcontext->bind_blend_state = grate_bind_blend_state;
   pcontext->delete_blend_state = grate_delete_blend_state;
}

static void *
grate_create_sampler_state(struct pipe_context *pcontext,
            const struct pipe_sampler_state *template)
{
   struct pipe_sampler_state *so = CALLOC_STRUCT(pipe_sampler_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_sampler_states(struct pipe_context *pcontext,
                          enum mesa_shader_stage shader,
                          unsigned start_slot, unsigned num_samplers,
                          void **samplers)
{
   struct grate_context *ctx = grate_context(pcontext);
   assert(shader == MESA_SHADER_FRAGMENT);
   assert(start_slot + num_samplers <= 16);
   for (unsigned i = 0; i < num_samplers; i++)
      ctx->samplers[start_slot + i] = samplers ? samplers[i] : NULL;
}

static void
grate_delete_sampler_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static struct pipe_sampler_view *
grate_create_sampler_view(struct pipe_context *pcontext,
                          struct pipe_resource *resource,
                          const struct pipe_sampler_view *template)
{
   struct pipe_sampler_view *so = CALLOC_STRUCT(pipe_sampler_view);
   if (!so)
      return NULL;

   *so = *template;
   so->texture = NULL;
   pipe_resource_reference(&so->texture, resource);
   pipe_reference_init(&so->reference, 1);
   so->context = pcontext;

   return so;
}

static void
grate_sampler_view_destroy(struct pipe_context *pcontext,
                           struct pipe_sampler_view *pview)
{
   pipe_resource_reference(&pview->texture, NULL);
   FREE(pview);
}

void
grate_context_sampler_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_sampler_state = grate_create_sampler_state;
   pcontext->bind_sampler_states = grate_bind_sampler_states;
   pcontext->delete_sampler_state = grate_delete_sampler_state;
   pcontext->create_sampler_view = grate_create_sampler_view;
   pcontext->sampler_view_destroy = grate_sampler_view_destroy;
   pcontext->sampler_view_release = u_default_sampler_view_release;
}

static int
grate_cull_face(int cull_face, bool front_ccw)
{
   switch (cull_face) {
   case PIPE_FACE_NONE:
      return TGR3D_CULL_NONE;

   case PIPE_FACE_FRONT:
      return front_ccw ? TGR3D_CULL_POS : TGR3D_CULL_NEG;

   case PIPE_FACE_BACK:
      return front_ccw ? TGR3D_CULL_NEG : TGR3D_CULL_POS;

   case PIPE_FACE_FRONT_AND_BACK:
      return TGR3D_CULL_BOTH;

   default:
      UNREACHABLE("unknown cull_face");
   }
}

static void *
grate_create_rasterizer_state(struct pipe_context *pcontext,
                              const struct pipe_rasterizer_state *template)
{
   struct grate_rasterizer_state *so = CALLOC_STRUCT(grate_rasterizer_state);
   if (!so)
      return NULL;

   so->base = *template;

   so->draw_params = TGR3D_IDX_SET_PRIM_FLAT_VTX(!template->flatshade_first);

   /* normal */
   so->cull_face[0] = TGR3D_SU_PARAM_FRONT_FACE(!template->front_ccw);
   so->cull_face[0] |= TGR3D_SU_PARAM_CULL(grate_cull_face(template->cull_face,
                                           template->front_ccw));

   /* y-inverted */
   so->cull_face[1] = TGR3D_SU_PARAM_FRONT_FACE(template->front_ccw);
   so->cull_face[1] |= TGR3D_SU_PARAM_CULL(grate_cull_face(template->cull_face,
                                           !template->front_ccw));

   return so;
}

static void
grate_bind_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->rast = so;
}

static void
grate_delete_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_rasterizer_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_rasterizer_state = grate_create_rasterizer_state;
   pcontext->bind_rasterizer_state = grate_bind_rasterizer_state;
   pcontext->delete_rasterizer_state = grate_delete_rasterizer_state;
}

static int
grate_compare_func(enum pipe_compare_func func)
{
   switch (func) {
   case PIPE_FUNC_NEVER: return TGR3D_FUNC_NEVER;
   case PIPE_FUNC_LESS: return TGR3D_FUNC_LESS;
   case PIPE_FUNC_EQUAL: return TGR3D_FUNC_EQUAL;
   case PIPE_FUNC_LEQUAL: return TGR3D_FUNC_LEQUAL;
   case PIPE_FUNC_GREATER: return TGR3D_FUNC_GREATER;
   case PIPE_FUNC_NOTEQUAL: return TGR3D_FUNC_NOTEQUAL;
   case PIPE_FUNC_ALWAYS: return TGR3D_FUNC_ALWAYS;
   default: UNREACHABLE("unknown pipe_compare_func");
   }
}

static void *
grate_create_zsa_state(struct pipe_context *pcontext,
                       const struct pipe_depth_stencil_alpha_state *template)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_zsa_state *so = CALLOC_STRUCT(grate_zsa_state);
   grate_trace();
   if (!so)
      return NULL;

   so->base = *template;

   uint32_t depth_test = 0;
   depth_test |= TGR3D_QR_Z_TEST_Z_FUNC(grate_compare_func(template->depth_func));
   depth_test |= TGR3D_QR_Z_TEST_Z_ENABLE(template->depth_enabled);
   depth_test |= TGR3D_QR_Z_TEST_QRAST_FB_WRITE(template->depth_writemask);
   depth_test |= TGR3D_QR_Z_TEST_Z_CLAMP(TGR3D_Z_CLAMP_KILL);

   so->commands[0] = host1x_opcode_incr(REG_TGR3D_QR_Z_TEST, 1);
   so->commands[1] = depth_test;
   so->num_commands = 2;

   if (context->drm->soc_id == DRM_TEGRA_SOC_T114) {
      so->commands[2] = host1x_opcode_incr(0xe45, 1);
      so->commands[3] = depth_test;
      so->num_commands = 4;
   }

   return so;
}

static void
grate_bind_zsa_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->zsa = so;
}

static void
grate_delete_zsa_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_zsa_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_depth_stencil_alpha_state = grate_create_zsa_state;
   pcontext->bind_depth_stencil_alpha_state = grate_bind_zsa_state;
   pcontext->delete_depth_stencil_alpha_state = grate_delete_zsa_state;
}

/*
 * Note: this does not include the stride, which needs to be mixed in later
 **/
static uint32_t
attrib_mode(const struct pipe_vertex_element *e)
{
   const struct util_format_description *desc = util_format_description(e->src_format);
   const int c = util_format_get_first_non_void_channel(e->src_format);
   uint32_t type, format;

   assert(!desc->is_mixed);
   assert(c >= 0);

   switch (desc->channel[c].type) {
   case UTIL_FORMAT_TYPE_UNSIGNED:
   case UTIL_FORMAT_TYPE_SIGNED:
      switch (desc->channel[c].size) {
      case 8:
         type = TGR3D_ATTR_FMT_U8;
         break;

      case 16:
         type = TGR3D_ATTR_FMT_U16;
         break;

      case 32:
         type = TGR3D_ATTR_FMT_U32;
         break;

      default:
         UNREACHABLE("invalid channel-size");
      }

      if (desc->channel[c].type == UTIL_FORMAT_TYPE_SIGNED)
         type += 2;

      if (desc->channel[c].normalized)
         type += 1;

      break;

   case UTIL_FORMAT_TYPE_FIXED:
      assert(desc->channel[c].size == 32);
      type = TGR3D_ATTR_FMT_X32;
      break;

   case UTIL_FORMAT_TYPE_FLOAT:
      assert(desc->channel[c].size == 32); /* TODO: float16 ? */
      type = TGR3D_ATTR_FMT_F32;
      break;

   default:
      UNREACHABLE("invalid channel-type");
   }

   format  = TGR3D_IDX_ATTRIBUTE_MODE_ATTR_FMT(type);
   format |= TGR3D_IDX_ATTRIBUTE_MODE_ATTR_SIZE(desc->nr_channels);
   return format;
}

static void *
grate_create_vertex_state(struct pipe_context *pcontext, unsigned int count,
                          const struct pipe_vertex_element *elements)
{
   unsigned int i;
   uint16_t mask = 0;
   struct grate_vertex_state *vtx = CALLOC_STRUCT(grate_vertex_state);
   if (!vtx)
      return NULL;

   for (i = 0; i < count; ++i) {
      const struct pipe_vertex_element *src = elements + i;
      struct grate_vertex_element *dst = vtx->elements + i;
      dst->attrib = attrib_mode(src);
      dst->buffer_index = src->vertex_buffer_index;
      dst->offset = src->src_offset;
      dst->stride = src->src_stride;
      mask |= 1 << i;
   }

   vtx->num_elements = count;
   vtx->mask = mask;

   return vtx;
}

static void
grate_bind_vertex_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->vs = so;
}

static void
grate_delete_vertex_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static void
emit_attribs(struct grate_context *context, uint32_t **ptrp)
{
   unsigned int i;
   struct grate_stream *stream = &context->gr3d->stream;

   assert(context->vs);

   for (i = 0; i < context->vs->num_elements; ++i) {
      const struct pipe_vertex_buffer *vb;
      const struct grate_vertex_element *e = context->vs->elements + i;
      const struct grate_resource *r;

      assert(e->buffer_index < context->vbs.count);
      vb = context->vbs.vb + e->buffer_index;
      assert(!vb->is_user_buffer);
      r = grate_resource(vb->buffer.resource);

      uint32_t attrib = e->attrib;
      assert(e->stride < 1 << 24);
      attrib |= TGR3D_IDX_ATTRIBUTE_MODE_ATTR_STRIDE(e->stride);

      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_IDX_ATTRIBUTE_BASE(i), 2));
      grate_stream_push_reloc(stream, ptrp, r->bo, vb->buffer_offset + e->offset);
      GRATE_PUSHBUF_WORD(*ptrp, attrib);
   }
}

static void
emit_render_targets(struct grate_context *context, uint32_t **ptrp)
{
   unsigned int i;
   struct grate_stream *stream = &context->gr3d->stream;
   const struct grate_framebuffer_state *fb = &context->framebuffer;

   /* Emit only populated slots: slot 0 is absent for colour-only draws,
    * and Gallium may also leave holes among the colour attachments.
    */
   for (i = 0; i < fb->num_rts; ++i) {
      if (!(fb->rt_mask & (1u << i)))
         continue;

      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_GLOBAL_SURFDESC(i), 1));
      GRATE_PUSHBUF_WORD(*ptrp, fb->rt_params[i]);
      GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_GLOBAL_SURFADDR(i), 1));
      grate_stream_push_reloc(stream, ptrp, fb->rt_bos[i], 0);
   }

   uint32_t enable = fb->rt_mask;
   if (fb->base.zsbuf.texture && context->zsa && context->zsa->base.depth_enabled)
      enable |= TGR3D_DW_ST_ENABLE_Z_TEST(TGR3D_STATE_ENABLED);

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(REG_TGR3D_DW_ST_ENABLE, 1));
   GRATE_PUSHBUF_WORD(*ptrp, enable);
}

static void
emit_scissor(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, ptrp, context->no_scissor, 3, 0);
}

static void
emit_viewport(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, ptrp, context->viewport, 10, 0);
}

static void
emit_guardband(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, ptrp, context->guardband, 4, 0);
}

static void
emit_zsa_state(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, ptrp, context->zsa->commands, context->zsa->num_commands, 0);
}

static void
emit_vs_uniforms(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   struct pipe_constant_buffer *constbuf = &context->constant_buffer[MESA_SHADER_VERTEX];
   unsigned num_imm = context->vshader ? context->vshader->num_immediates : 0;
   unsigned user_words = 0;

   if (constbuf->user_buffer != NULL) {
      assert(constbuf->buffer_size % sizeof(uint32_t) == 0);
      user_words = constbuf->buffer_size / 4;
   }

   /* one block from slot 0: the shader's immediates, then the application's uniforms; see grate_compiler.h */
   unsigned len = num_imm * 4 + user_words;
   if (len == 0)
      return;

   assert(len <= 256 * 4);

   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_imm(REG_TGR3D_VPE_CONST_OFFSET, 0));
   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_nonincr(REG_TGR3D_VPE_CONST_DATA, len));
   if (num_imm)
      grate_stream_push_words(stream, ptrp, context->vshader->immediates[0], num_imm * 4, 0);
   if (user_words)
      grate_stream_push_words(stream, ptrp, constbuf->user_buffer, user_words, 0);
}

/* Tegra fragment constants are sign:1 exponent:6 mantissa:13 (bias 31).
 * Flush values below the normalized range rather than wrapping the exponent.
 */
static uint32_t
fragment_fp20(float value)
{
   uint32_t bits = fui(value);
   unsigned exponent = (bits >> 23) & 0xff;
   uint32_t sign = (bits >> 12) & 0x80000;
   if (exponent == 0xff)
      return sign | 0x7e000 | ((bits & 0x7fffff) ? 1 : 0);
   int adjusted = (int)exponent - 127 + 31;
   if (adjusted <= 0)
      return 0;
   if (adjusted >= 63)
      return sign | 0x7e000;
   return sign | ((uint32_t)adjusted << 13) | ((bits & 0x7fffff) >> 10);
}

static void
emit_fs_uniforms(struct grate_context *context, uint32_t **ptrp)
{
   const struct grate_fp_info *info = &context->fshader->info;
   const struct pipe_constant_buffer *cb = &context->constant_buffer[MESA_SHADER_FRAGMENT];
   if (!info->num_constants)
      return;
   GRATE_PUSHBUF_WORD(*ptrp, host1x_opcode_incr(0x820, info->num_constants));
   for (unsigned i = 0; i < info->num_constants; i++) {
      float value = info->constants[i].immediate;
      int word = info->constants[i].word;
      if (word >= 0) {
         assert(cb->user_buffer && ((unsigned)word + 1) * sizeof(float) <= cb->buffer_size);
         memcpy(&value, (const uint8_t *)cb->user_buffer + word * sizeof(float), sizeof(value));
      }
      GRATE_PUSHBUF_WORD(*ptrp, fragment_fp20(value));
   }
}

static void
emit_shader(struct grate_stream *stream, uint32_t **ptrp, struct grate_shader_blob *blob)
{
   grate_stream_push_words(stream, ptrp, blob->commands, blob->num_commands, 0);
}

static void
emit_program(struct grate_context *context, uint32_t **ptrp)
{
   struct grate_stream *stream = &context->gr3d->stream;
   uint32_t cull_face_linker_setup;

   emit_shader(stream, ptrp, &context->vshader->blob);
   emit_shader(stream, ptrp, &context->fshader->blob);

   cull_face_linker_setup = TGR3D_SU_PARAM_SUBPIX_XOFF(0x38) |
                            TGR3D_SU_PARAM_SUBPIX_YOFF(0x38) |
                            TGR3D_SU_PARAM_TRANSPOSE_XY(TGR3D_STATE_DISABLED) |
                            TGR3D_SU_PARAM_CLIP_ENABLE(TGR3D_STATE_ENABLED);

   /* depends on cull-face */
   cull_face_linker_setup |= context->rast->cull_face[context->y_invert];

   /* depends on linking */
   struct grate_fp_info *info = &context->fshader->info;
   cull_face_linker_setup |= TGR3D_SU_PARAM_LAST_INST(0 < info->num_inputs ?
                                                      (info->num_inputs - 1) : 0);

   uint32_t linker_insts[3 + info->num_inputs * 2];
   linker_insts[0] = host1x_opcode_incr(REG_TGR3D_SU_PARAM, 1);
   linker_insts[1] = cull_face_linker_setup;
   linker_insts[2] = host1x_opcode_incr(REG_TGR3D_SU_INST_EVEN(0), info->num_inputs * 2);

   for (int i = 0; i < info->num_inputs; ++i) {
      linker_insts[3 + i * 2] = info->inputs[i].src;
      linker_insts[3 + i * 2 + 1] = info->inputs[i].dst;
   }

   if (context->rast->base.flatshade && info->color_input >= 0)
      linker_insts[3 + info->color_input * 2 + 1] |= 0xf << 16;

   grate_stream_push_words(stream, ptrp, linker_insts, ARRAY_SIZE(linker_insts), 0);
}

/* Base-level linear 2D textures. The texture engine derives row stride from
 * width; sampler-only resources therefore have tightly packed rows.
 */
static void
emit_textures(struct grate_context *ctx, uint32_t **ptrp)
{
   struct grate_stream *stream = &ctx->gr3d->stream;
   uint32_t *ptr = *ptrp;
   for (unsigned i = 0; i < 16; i++) {
      struct pipe_sampler_view *view = ctx->sampler_views[i];
      struct pipe_sampler_state *sampler = ctx->samplers[i];
      if (!view || !sampler)
         continue;
      struct grate_resource *res = grate_resource(view->texture);
      assert(view->target == PIPE_TEXTURE_2D && view->u.tex.first_level == 0);
      assert(view->u.tex.last_level == 0 && sampler->min_mip_filter == PIPE_TEX_MIPFILTER_NONE);
      unsigned row_bytes = res->b.width0 * util_format_get_blocksize(res->b.format);
      /* POT layout is tightly packed. NPOT layout rounds rows to 64 bytes,
       * even when the logical dimensions themselves are powers of two.
       */
      bool pot_layout = util_is_power_of_two_or_zero(res->b.width0) &&
                        util_is_power_of_two_or_zero(res->b.height0) &&
                        (res->pitch == row_bytes || res->b.height0 == 1);
      unsigned native_pitch = pot_layout ? row_bytes : align(row_bytes, 64);
      bool native_layout = res->pitch == native_pitch ||
                           (res->b.height0 == 1 && res->pitch >= row_bytes);
      if (!native_layout)
         fprintf(stderr, "GRATE unsupported sampler layout: %s %ux%u pitch=%u native=%u\n",
                 util_format_name(res->b.format),res->b.width0,res->b.height0,res->pitch,native_pitch);
      assert(native_layout);
      if (grate_debug & GRATE_DEBUG_TRACE)
         fprintf(stderr, "GRATE sampler: %s view %s swizzle=%u,%u,%u,%u size=%ux%u pitch=%u format=%u\n",
                 util_format_name(res->b.format), util_format_name(view->format),
                 view->swizzle_r,view->swizzle_g,view->swizzle_b,view->swizzle_a,
                 res->b.width0,res->b.height0,res->pitch,res->format);
      /* Texture channels follow GL order, independently of the framebuffer
       * store's BGRA packing. Use the view's format for reinterpretation. */
      unsigned texture_format = grate_pixel_format(view->format);
      switch (view->format) {
      case PIPE_FORMAT_B8G8R8A8_UNORM:
         texture_format = TGR3D_SURF_FORMAT_B8G8R8A8;
         break;
      case PIPE_FORMAT_B8G8R8X8_UNORM:
         texture_format = TGR3D_SURF_FORMAT_B8G8R8X8;
         break;
      case PIPE_FORMAT_R8G8B8A8_UNORM:
         texture_format = TGR3D_SURF_FORMAT_R8G8B8A8;
         break;
      case PIPE_FORMAT_R8G8B8X8_UNORM:
         texture_format = TGR3D_SURF_FORMAT_R8G8B8X8;
         break;
      default:
         break;
      }
      uint32_t lo = TGR3D_TEX_TEXDESC_LO_SURF_FORMAT(texture_format);
      if (sampler->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_EDGE) lo |= 2;
      else assert(sampler->wrap_s == PIPE_TEX_WRAP_REPEAT);
      if (sampler->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_EDGE) lo |= 1;
      else assert(sampler->wrap_t == PIPE_TEX_WRAP_REPEAT);
      if (sampler->min_img_filter == PIPE_TEX_FILTER_LINEAR) lo |= TGR3D_TEX_TEXDESC_LO_LERP_MIN;
      if (sampler->mag_img_filter == PIPE_TEX_FILTER_LINEAR) lo |= TGR3D_TEX_TEXDESC_LO_LERP_MAG;
      uint32_t hi = 0x80; /* base level only */
      if (pot_layout) {
         hi |= TGR3D_TEX_TEXDESC_HI_LOG2_WIDTH(util_logbase2(res->b.width0));
         hi |= TGR3D_TEX_TEXDESC_HI_LOG2_HEIGHT(util_logbase2(res->b.height0));
      } else {
         hi |= 0x40 | TGR3D_TEX_TEXDESC_HI_WIDTH(res->b.width0) |
                      TGR3D_TEX_TEXDESC_HI_HEIGHT(res->b.height0);
      }
      GRATE_PUSHBUF_WORD(ptr, host1x_opcode_incr(REG_TGR3D_TEX_TEXADDR(i), 1));
      grate_stream_push_reloc(stream, &ptr, res->bo, 0);
      GRATE_PUSHBUF_WORD(ptr, host1x_opcode_incr(REG_TGR3D_TEX_TEXDESC_LO(i), 2));
      GRATE_PUSHBUF_WORD(ptr, lo);
      GRATE_PUSHBUF_WORD(ptr, hi);
   }
   *ptrp = ptr;
}

void
grate_emit_state(struct grate_context *context, uint32_t **ptrp)
{
   emit_render_targets(context, ptrp);
   emit_viewport(context, ptrp);
   emit_guardband(context, ptrp);
   emit_scissor(context, ptrp);
   emit_zsa_state(context, ptrp);
   emit_attribs(context, ptrp);
   emit_vs_uniforms(context, ptrp);
   emit_fs_uniforms(context, ptrp);
   emit_program(context, ptrp);
   emit_textures(context, ptrp);
}

void
grate_context_vbo_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_vertex_elements_state = grate_create_vertex_state;
   pcontext->bind_vertex_elements_state = grate_bind_vertex_state;
   pcontext->delete_vertex_elements_state = grate_delete_vertex_state;
}
