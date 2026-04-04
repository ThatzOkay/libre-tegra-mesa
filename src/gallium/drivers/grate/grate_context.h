#ifndef GRATE_CONTEXT_H
#define GRATE_CONTEXT_H

#include "util/slab.h"

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include "grate_common.h"
#include "grate_device.h"
#include "grate_state.h"
#include "grate_stream.h"

struct grate_framebuffer_state {
   struct pipe_framebuffer_state base;
   int num_rts;
   struct grate_bo *rt_bos[TGR3D_MAX_RENDER_TARGETS];
   uint32_t rt_params[TGR3D_MAX_RENDER_TARGETS];
   uint32_t rt_mask;
};

struct grate_channel {
   struct grate_context *context;
   struct grate_stream stream;
};

struct grate_context {
   struct pipe_context base;

   struct grate_device *drm;
   struct grate_channel *gr2d;
   struct grate_channel *gr3d;

   struct grate_framebuffer_state framebuffer;

   struct slab_child_pool transfer_pool;

   struct grate_vertex_state *vs;
   struct grate_vertexbuf_state vbs;
   struct pipe_constant_buffer constant_buffer[PIPE_MAX_CONSTANT_BUFFERS]; // ??, stolen from other drivers but no idea

   struct grate_zsa_state *zsa;
   struct grate_rasterizer_state *rast;

   struct grate_vertex_shader_state *vshader;
   struct grate_fragment_shader_state *fshader;

   uint32_t no_scissor[3];
   uint32_t viewport[10];
   uint32_t guardband[4];
   bool y_invert;
};

static inline struct grate_context *
grate_context(struct pipe_context *context)
{
   return (struct grate_context *)context;
}

struct pipe_context *
grate_screen_context_create(struct pipe_screen *pscreen,
                            void *priv, unsigned flags);

#endif
