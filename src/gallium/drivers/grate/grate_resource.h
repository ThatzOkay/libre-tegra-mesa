#ifndef GRATE_RESOURCE_H
#define GRATE_RESOURCE_H

#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_transfer.h"

struct grate_resource {
   struct pipe_resource b;
   struct grate_bo *bo;
   /* Linear sampler copy for imported buffers with a non-native stride. */
   struct grate_bo *sampler_bo;
   unsigned int pitch;
   unsigned int level_pitch[PIPE_MAX_TEXTURE_LEVELS];
   unsigned int level_offset[PIPE_MAX_TEXTURE_LEVELS];
   unsigned int level_height[PIPE_MAX_TEXTURE_LEVELS];
   unsigned int tiled : 1;
   unsigned int format : 5;
   /* Z24 depth/stencil formats: the GPU has no 24-bit depth. The depth plane
    * is 16-bit non-linear (Z16NL, 2 bytes per pixel, rows `pitch` apart); a
    * stencil format adds an 8-bit plane after it (stencil_offset bytes from
    * the start of the buffer, rows `stencil_pitch` apart), which the GPU
    * binds as its own surface. Both are 0 for every other format.
    */
   unsigned int stencil_offset;
   unsigned int stencil_pitch;
};

struct grate_transfer {
   struct pipe_transfer b;
};

static inline struct grate_resource *
grate_resource(struct pipe_resource *resource)
{
   return (struct grate_resource *)resource;
}

int
grate_pixel_format(enum pipe_format format);

/* A 24-bit depth format that is stored as Z16NL (+ a separate S8 plane when
 * it has stencil). */
bool grate_format_is_emulated_z24(enum pipe_format format);

/* The 16-bit value Z16NL stores for a depth in [0, 1]. */
uint16_t grate_pack_z16nl(double depth);

void
grate_context_resource_init(struct pipe_context *pcontext);

void
grate_screen_resource_init(struct pipe_screen *pscreen);

#endif
