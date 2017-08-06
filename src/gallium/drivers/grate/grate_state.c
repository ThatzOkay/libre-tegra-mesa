#include <stdio.h>

#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_state.h"


static void
grate_set_sample_mask(struct pipe_context *pcontext,
                      unsigned int sample_mask)
{
   unimplemented();
}

static void
grate_set_constant_buffer(struct pipe_context *pcontext, enum pipe_shader_type shader,
                          uint index, bool take_ownership,
                          const struct pipe_constant_buffer *buffer)
{
   unimplemented();
}

static void
grate_set_framebuffer_state(struct pipe_context *pcontext,
                            const struct pipe_framebuffer_state *framebuffer)
{
   struct grate_context *context = grate_context(pcontext);
   unsigned int i;

   for (i = 0; i < framebuffer->nr_cbufs; i++) {
      struct pipe_surface *ref = framebuffer->cbufs[i];

      if (i >= framebuffer->nr_cbufs)
         ref = NULL;

      pipe_surface_reference(&context->framebuffer.base.cbufs[i],
                   ref);
   }

   pipe_surface_reference(&context->framebuffer.base.zsbuf,
                framebuffer->zsbuf);

   context->framebuffer.base.width = framebuffer->width;
   context->framebuffer.base.height = framebuffer->height;
   context->framebuffer.base.nr_cbufs = framebuffer->nr_cbufs;
}

static void
grate_set_polygon_stipple(struct pipe_context *pcontext,
                          const struct pipe_poly_stipple *stipple)
{
   unimplemented();
}

static void
grate_set_scissor_states(struct pipe_context *pcontext,
                         unsigned start_slot,
                         unsigned num_scissors,
                         const struct pipe_scissor_state * scissors)
{
   assert(num_scissors == 1);
   unimplemented();
}

static void
grate_set_viewport_states(struct pipe_context *pcontext,
                          unsigned start_slot,
                          unsigned num_viewports,
                          const struct pipe_viewport_state *viewports)
{
   assert(num_viewports == 1);
   unimplemented();
}

static void
grate_set_vertex_buffers(struct pipe_context *pcontext,
                         unsigned count,
                         const struct pipe_vertex_buffer *buffer)
{
   unimplemented();
}


static void
grate_set_sampler_views(struct pipe_context *pctx, enum pipe_shader_type shader,
                        unsigned start_slot, unsigned num_views,
                        unsigned unbind_num_trailing_slots,
                        bool take_ownership,
                        struct pipe_sampler_view **views)
{
   unimplemented();
}

static void
grate_set_blend_color(struct pipe_context *pctx,
                      const struct pipe_blend_color *blend_color)
{
   unimplemented();
}

static void
grate_set_stencil_ref(struct pipe_context *pctx,
                      const struct pipe_stencil_ref stencil_ref)
{
   unimplemented();
}

static void
grate_fp_state_bind(struct pipe_context *pctx, void *hwcso)
{
   unimplemented();
}

static void
grate_vp_state_bind(struct pipe_context *pctx, void *hwcso)
{
   unimplemented();
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
   unimplemented();
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
                          enum pipe_shader_type shader,
                          unsigned start_slot, unsigned num_samplers,
                          void **samplers)
{
   unimplemented();
}

static void
grate_delete_sampler_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_sampler_init(struct pipe_context *pcontext)
{
   pcontext->create_sampler_state = grate_create_sampler_state;
   pcontext->bind_sampler_states = grate_bind_sampler_states;
   pcontext->delete_sampler_state = grate_delete_sampler_state;
}

static void *
grate_create_rasterizer_state(struct pipe_context *pcontext,
                              const struct pipe_rasterizer_state *template)
{
   struct pipe_rasterizer_state *so = CALLOC_STRUCT(pipe_rasterizer_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   unimplemented();
}

static void
grate_delete_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_rasterizer_init(struct pipe_context *pcontext)
{
   pcontext->create_rasterizer_state = grate_create_rasterizer_state;
   pcontext->bind_rasterizer_state = grate_bind_rasterizer_state;
   pcontext->delete_rasterizer_state = grate_delete_rasterizer_state;
}

static void *
grate_create_zsa_state(struct pipe_context *pcontext,
                       const struct pipe_depth_stencil_alpha_state *template)
{
   struct pipe_depth_stencil_alpha_state *so = CALLOC_STRUCT(pipe_depth_stencil_alpha_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_zsa_state(struct pipe_context *pcontext, void *so)
{
   unimplemented();
}

static void
grate_delete_zsa_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_zsa_init(struct pipe_context *pcontext)
{
   pcontext->create_depth_stencil_alpha_state = grate_create_zsa_state;
   pcontext->bind_depth_stencil_alpha_state = grate_bind_zsa_state;
   pcontext->delete_depth_stencil_alpha_state = grate_delete_zsa_state;
}

static void *
grate_create_vertex_state(struct pipe_context *pcontext, unsigned int count,
                          const struct pipe_vertex_element *elements)
{
   unimplemented();
   return NULL;
}

static void
grate_bind_vertex_state(struct pipe_context *pcontext, void *so)
{
   unimplemented();
}

static void
grate_delete_vertex_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_vbo_init(struct pipe_context *pcontext)
{
   pcontext->create_vertex_elements_state = grate_create_vertex_state;
   pcontext->bind_vertex_elements_state = grate_bind_vertex_state;
   pcontext->delete_vertex_elements_state = grate_delete_vertex_state;
}
