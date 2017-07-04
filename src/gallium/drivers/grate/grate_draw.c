#include <stdio.h>

#include "pipe/p_state.h"

#include "util/u_draw.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_draw.h"
#include "grate_program.h"
#include "grate_state.h"

#include "tgr_3d.xml.h"
#include "host1x01_hardware.h"

static int
grate_primitive_type(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return TGR3D_PRIMITIVE_TYPE_POINTS;

   case MESA_PRIM_LINES:
      return TGR3D_PRIMITIVE_TYPE_LINES;

   case MESA_PRIM_LINE_LOOP:
      return TGR3D_PRIMITIVE_TYPE_LINE_LOOP;

   case MESA_PRIM_LINE_STRIP:
      return TGR3D_PRIMITIVE_TYPE_LINE_STRIP;

   case MESA_PRIM_TRIANGLES:
      return TGR3D_PRIMITIVE_TYPE_TRIANGLES;

   case MESA_PRIM_TRIANGLE_STRIP:
      return TGR3D_PRIMITIVE_TYPE_TRIANGLE_STRIP;

   case MESA_PRIM_TRIANGLE_FAN:
      return TGR3D_PRIMITIVE_TYPE_TRIANGLE_FAN;

   default:
      unreachable("unexpected enum pipe_prim_type");
   }
}

static void
grate_draw_vbo(struct pipe_context *pcontext,
               const struct pipe_draw_info *info,
               unsigned drawid_offset,
               const struct pipe_draw_indirect_info *indirect,
               const struct pipe_draw_start_count_bias *draws,
               unsigned num_draws)
{
   int err;
   uint32_t value;
   struct grate_context *context = grate_context(pcontext);
   struct grate_stream *stream = &context->gr3d->stream;

   if (num_draws > 1) {
      util_draw_multi(pcontext, info, drawid_offset, indirect, draws, num_draws);
      return;
   }

   err = grate_stream_begin(stream);
   if (err < 0) {
      fprintf(stderr, "grate_stream_begin() failed: %d\n", err);
      return;
   }

   grate_stream_push_setclass(stream, HOST1X_CLASS_GR3D);

   grate_emit_state(context);

   assert(!info->index_size);

   uint16_t out_mask = context->vshader->output_mask;
   grate_stream_push(stream, host1x_opcode_incr(TGR3D_VP_ATTRIB_IN_OUT_SELECT, 1));
   grate_stream_push(stream, ((uint32_t)context->vs->mask << 16) | out_mask);

   /* draw params */
   value  = TGR3D_VAL(DRAW_PARAMS, INDEX_MODE, TGR3D_INDEX_MODE_NONE);
   value |= context->rast->draw_params;
   value |= TGR3D_VAL(DRAW_PARAMS, PRIMITIVE_TYPE, grate_primitive_type(info->mode));
   value |= TGR3D_VAL(DRAW_PARAMS, FIRST, draws[0].start);
   value |= 0xC0000000; /* flush input caches? */

   grate_stream_push(stream, host1x_opcode_incr(TGR3D_DRAW_PARAMS, 1));
   grate_stream_push(stream, value);

   unsigned count = draws[0].count;
   assert(count > 0 && count < (1 << 11));
   value  = TGR3D_VAL(DRAW_PRIMITIVES, INDEX_COUNT, count - 1);
   value |= TGR3D_VAL(DRAW_PRIMITIVES, OFFSET, draws[0].start);
   grate_stream_push(stream, host1x_opcode_incr(TGR3D_DRAW_PRIMITIVES, 1));
   grate_stream_push(stream, value);

   grate_stream_end(stream);

   grate_stream_flush(stream);
}

void
grate_context_draw_init(struct pipe_context *pcontext)
{
   pcontext->draw_vbo = grate_draw_vbo;
}
