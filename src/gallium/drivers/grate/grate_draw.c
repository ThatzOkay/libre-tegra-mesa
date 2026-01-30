#include <stdio.h>
#include <math.h>

#include "pipe/p_state.h"
#include "util/u_bitcast.h"
#include "util/u_draw.h"
#include "util/u_helpers.h"
#include "util/u_prim.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_draw.h"
#include "grate_program.h"
#include "grate_resource.h"
#include "grate_state.h"
#include "grate_device.h"

#include "tgr_3d.xml.h"
#include "host1x01_hardware.h"

static int
grate_primitive_type(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return TGR3D_PRIM_TYPE_POINTS;

   case MESA_PRIM_LINES:
      return TGR3D_PRIM_TYPE_LINES;

   case MESA_PRIM_LINE_LOOP:
      return TGR3D_PRIM_TYPE_LINE_LOOP;

   case MESA_PRIM_LINE_STRIP:
      return TGR3D_PRIM_TYPE_LINE_STRIP;

   case MESA_PRIM_TRIANGLES:
      return TGR3D_PRIM_TYPE_TRIS;

   case MESA_PRIM_TRIANGLE_STRIP:
      return TGR3D_PRIM_TYPE_TRI_STRIP;

   case MESA_PRIM_TRIANGLE_FAN:
      return TGR3D_PRIM_TYPE_TRI_FAN;

   default:
      UNREACHABLE("unexpected enum pipe_prim_type");
   }
}

static int
grate_init_state(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;
   enum drm_tegra_soc_id soc_id = context->drm->soc_id;

   /* Tegra114 specific stuff */
   grate_stream_push(stream, host1x_opcode_imm(0xe44, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0x807, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc00, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc01, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc02, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc03, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc30, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc31, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc32, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc33, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc40, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc41, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc42, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc43, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc50, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc51, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc52, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xc53, 0x0000));

   grate_stream_push(stream, host1x_opcode_incr(0xe70, 0x0010));
   for (int i = 0; i < 16; i++)
      grate_stream_push(stream, 0x00000000);

   grate_stream_push(stream, host1x_opcode_imm(0xe80, 0x0f00));
   grate_stream_push(stream, host1x_opcode_imm(0xe84, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xe85, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xe86, 0x0000));
   grate_stream_push(stream, host1x_opcode_imm(0xe87, 0x0000));

   /* Tegra30 specific stuff */
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_TIMESTAMP_CTL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_TIMESTAMP_LOW, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_TIMESTAMP_HIGH, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_PIXEL_COUNT_CTRL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_PIXEL_COUNT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_WRITE_MASK,
                                               TGR3D_GSHIM_WRITE_MASK_GPU_A(TGR3D_STATE_ENABLED) |
                                               TGR3D_GSHIM_WRITE_MASK_GPU_B(TGR3D_STATE_ENABLED)));

   /*
    * 0x75x should be written after 0xb00, otherwise non-pow2
    * textures are corrupted. Reason is unknown. Looks like
    * combination of 0x75x register bits affects the texture size.
    *
    * The 0x75x registers contain garbage after machine's power-off,
    * but values are retained on soft reboot.
    */
   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_TEX_TEXDESC_NPOT_AUX, 0x10));
   for (int i = 0; i < 16; i++)
      grate_stream_push(stream, 0x00000000);

   /*
    * Tegra114 has additional texture descriptors. The order may
    * be important,hence it's placed in a middle of T30 regs until
    * we'll know that this is unnecessary.
    */
   grate_stream_push(stream, host1x_opcode_incr(0x770, 0x0030)); // TODO: no reg
   for (int i = 0; i < 16 + 2 * 16; i++)
      grate_stream_push(stream, 0x00000000);

   grate_stream_push(stream, host1x_opcode_imm(0x7e0, 0x0001)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(0x7e1, 0x0000)); // TODO: no reg

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_READ_SELECT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_ENABLE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_STALL, 0));
   grate_stream_push(stream, host1x_opcode_imm(0xb07, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_WAIT, 0));
   grate_stream_push(stream, host1x_opcode_imm(0xb09, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_COMB, 0));
   grate_stream_push(stream, host1x_opcode_imm(0xb0b, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_HWR_WAIT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_HWR_XFER, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_SYNCPT_WAIT, 0));
   grate_stream_push(stream, host1x_opcode_imm(0xb0f, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DLB_CONTROL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DLB_RANGE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DLB_TRIGGER, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DEBUG0, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_MEMORY_OUTPUT_READS, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_HORIZONTAL_SWATH_RENDERING, 0));

   /* Common stuff */
   grate_stream_push(stream, host1x_opcode_imm(0x00d, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_CLK_COUNT, 0));
   grate_stream_push(stream, host1x_opcode_imm(0x00f, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_XFER_COUNT, 0));
   grate_stream_push(stream, host1x_opcode_imm(0x011, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_WAIT_COUNT, 0));
   grate_stream_push(stream, host1x_opcode_imm(0x013, 0)); // TODO: no reg
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_EN_COUNT, 0));
   grate_stream_push(stream, host1x_opcode_imm(0x015, 0)); // TODO: no reg

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_IDX_ATTR_MASK, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_IDX_SET_PRIM, 0));

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_IDX_IDX_CTL,
                                               TGR3D_IDX_IDX_CTL_VAR_IBUF_SIZE |
                                               TGR3D_IDX_IDX_CTL_VAR_OBUF_SIZE |
                                               TGR3D_IDX_IDX_CTL_LATE_BINDING));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_IDX_IDX_STAT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_IDX_NV_MCCIF_FIFOCTRL_RO, 0));

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_VPE_MODE, 5));
   // TODO: document/convert these values
   grate_stream_push(stream, 0x00000011); /* REG_TGR3D_VPE_MODE */
   grate_stream_push(stream, 0x0000ffff);
   grate_stream_push(stream, 0x00ff0000);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_VPE_GEOM_STALL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_VPE_VPE_CTRL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_VPE_VPE_DEBUG,
                                               TGR3D_VPE_VPE_DEBUG_VPE_DEBUG_IBUF_SIZE |
                                               TGR3D_VPE_VPE_DEBUG_VPE_DEBUG_OBUF_SIZE));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_INST_EVEN(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_INST_EVEN(1), 0));

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_SU_PARAM, 25));
   // TODO: document/convert these values
   grate_stream_push(stream, 0xb8e00000); /* REG_TGR3D_SU_PARAM */
   grate_stream_push(stream, 0x00000000); /* TGR3D_POLYGON_OFFSET_UNITS */
   grate_stream_push(stream, 0x00000000); /* TGR3D_POLYGON_OFFSET_FACTOR */
   grate_stream_push(stream, 0x00000105); /* TGR3D_POINT_PARAMS */
   grate_stream_push(stream, u_bitcast_f2u(0.5f)); /* TGR3D_POINT_SIZE */
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); /* TGR3D_POIN_COORD_RANGE_MAX_S */
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); /* TGR3D_POIN_COORD_RANGE_MAX_T */
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); /* TGR3D_POIN_COORD_RANGE_MIN_S */
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); /* TGR3D_POIN_COORD_RANGE_MIN_T */
   grate_stream_push(stream, 0x00000000); /* TGR2D_LINE_PARAMS */
   grate_stream_push(stream, u_bitcast_f2u(0.5f)); /* TGR3D_HALF_LINE_WIDTH */
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); /* 0x34e - unknonwn */
   grate_stream_push(stream, 0x00000000); /* 0x34f - unknown */
   grate_stream_push(stream, 0x00000000); /* TGR3D_SCISSOR_HORIZ */
   grate_stream_push(stream, 0x00000000); /* TGR3D_SCISSOR_VERT */
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); /* TGR3D_VIEWPORT_X_BIAS */
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); /* TGR3D_VIEWPORT_Y_BIAS */
   grate_stream_push(stream, u_bitcast_f2u(0.5f - powf(2.0, -21))); /* TGR3D_VIEWPORT_Z_BIAS */
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); /* TGR3D_VIEWPORT_X_SCALE */
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); /* TGR3D_VIEWPORT_Y_SCALE */
   grate_stream_push(stream, u_bitcast_f2u(0.5f - powf(2.0, -21))); /* TGR3D_VIEWPORT_Z_SCALE */
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); /* TGR3D_GUARDBAND_WIDTH */
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); /* TGR3D_GUARDBAND_HEIGHT */
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); /* TGR3D_GUARDBAND_DEPTH */
   grate_stream_push(stream, 0x00000205); /* 0x35b - unknown */

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_CLKEN_OVERRIDE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_CLIP_CLKEN_OVERRIDE, 0));

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_QR_S_TEST,
                                               TGR3D_QR_S_TEST_S_MASK(0xff) |
                                               TGR3D_QR_S_TEST_S_FUNC(0x07)));
   grate_stream_push(stream, host1x_opcode_imm(TGR3D_STENCIL_BACK1, 0x07ff)); // TODO: no reg

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_QR_S_CTRL,
                                                TGR3D_QR_S_CTRL_COVERAGE_MERGE(TGR3D_COVERAGE_MERGE_XOR) |
                                                TGR3D_QR_S_CTRL_S_SURF_PTR(4)));
   grate_stream_push(stream, 0x00000040); /* TGR3D_STENCIL_PARAMS */
   // TODO: document/convert these values
   grate_stream_push(stream, 0x00000040); /* REG_TGR3D_QR_S_CTRL */
   grate_stream_push(stream, 0x00000310); /* TGR3D_DEPTH_TEST_PARAMS*/
   grate_stream_push(stream, 0x00000000); /* TGR3D_DEPTH_RANGE_NEAR */
   grate_stream_push(stream, 0x000fffff); /* TGR3D_DEPTH_RANGE_FAR */
   grate_stream_push(stream, 0x00000001); /* 0x406 - unknown */
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x1fff1fff);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000006);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000008);
   grate_stream_push(stream, 0x00000048);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_FLUSH, 0));

   if (soc_id == DRM_TEGRA_SOC_T114)
      grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_CTL, (0x2200 << 16) |
                                                                      TGR3D_PSEQ_CTL_MERGE_SPAN_STARTS |
                                                                      TGR3D_PSEQ_CTL_MERGE_REGISTERS |
                                                                      TGR3D_PSEQ_CTL_REMOVE_KILLED_PIXELS));
   else
      grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_CTL, TGR3D_PSEQ_CTL_MERGE_SPAN_STARTS |
                                                                      TGR3D_PSEQ_CTL_MERGE_REGISTERS |
                                                                      TGR3D_PSEQ_CTL_REMOVE_KILLED_PIXELS));

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_TIMEOUT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_PC, 0));

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_PSEQ_COMMAND_EVEN(0), 32));
   for (int i = 0; i < 32; i++)
      grate_stream_push(stream, 0);

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_INST_OFFSET, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_DBG_X, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_DBG_Y, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_DBG_CTL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_QUAD_ID, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_PSEQ_DWR_IF_STATE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_AT_CLKEN_OVERRIDE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_TEX_COLORKEY, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_TEX_TEXCTL,
                                               TGR3D_TEX_TEXCTL_TEXTURE_CACHE_EN(TGR3D_STATE_ENABLED)));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_TEX_CLKEN_OVERRIDE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_TEX_NV_MCCIF_FIFOCTRL_RO, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_LOGIC_OP, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_DW_ST_ENABLE, 0));

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_FDC_CONTROL, 13));
   // TODO: document/convert these values
   grate_stream_push(stream, 0x00000e00); /* REG_TGR3D_FDC_CONTROL */
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x000001ff);
   grate_stream_push(stream, 0x000001ff);
   grate_stream_push(stream, 0x000001ff);
   grate_stream_push(stream, 0x00000030);
   grate_stream_push(stream, 0x00000020);
   grate_stream_push(stream, 0x000001ff);
   grate_stream_push(stream, 0x00000100);
   grate_stream_push(stream, 0x0f0f0f0f);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);
   grate_stream_push(stream, 0x00000000);

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_PIX_ATTR, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_TRI_ATTR, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_INST_OFFSET, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_INSTRUMENT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_DITHER_TABLE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_FLUSH, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_S_OPERATION, 0));
   grate_stream_push(stream, host1x_opcode_imm(0xe29, 0)); // TODO: no reg

   if (soc_id == DRM_TEGRA_SOC_T114) {
      grate_stream_push(stream, host1x_opcode_imm(0x41a, 0xa00));
      grate_stream_push(stream, host1x_opcode_imm(0x416, 0x140));
   }

   return 0;
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

   if (!indirect && (!draws[0].count || !info->instance_count))
      return;

   err = grate_stream_begin(stream);
   if (err < 0) {
      fprintf(stderr, "%s: grate_stream_begin() failed: %d\n", __func__, err);
      return;
   }

   /*
    * The state needs to be re-initialized on each draw since tegra doesn't
    * support context switching in a good way, hardware is optimized for
    * uploading.
    */
   grate_init_state(context);
   grate_emit_state(context);

   uint16_t out_mask = context->vshader->output_mask;
   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_IDX_ATTR_MASK, 1));
   grate_stream_push(stream, ((uint32_t)context->vs->mask << 16) | out_mask);

   struct pipe_resource *index_buffer = NULL;
   unsigned offset = 0;
   if (info->index_size > 0) {
      unsigned index_offset = 0;
      if (info->has_user_indices) {
         if (!util_upload_index_buffer(pcontext, info, draws, &index_buffer, &index_offset, 64)) {
            fprintf(stderr, "%s: util_upload_index_buffer() failed\n", __func__);
            return;
         }
      } else
         index_buffer = info->index.resource;

      index_offset += draws->start * info->index_size;
      grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_IDX_INDEX_BASE, 1));
      grate_stream_push_reloc(stream, grate_resource(index_buffer)->bo, index_offset);
   } else
      offset = draws->start;

   unsigned index_size;
   switch (info->index_size) {
   case 0:
      index_size = 0;
      break;

   case 1:
      index_size = 1;
      break;

   case 2:
      index_size = 2;
      break;

   case 4:
      index_size = 3;
      break;

   default:
      UNREACHABLE("invalid index_size");
   }

   /* draw params */
   value  = TGR3D_IDX_SET_PRIM_DRAW_MODE(index_size);
   value |= context->rast->draw_params;
   value |= TGR3D_IDX_SET_PRIM_PRIM_TYPE(grate_primitive_type(info->mode));
   value |= TGR3D_IDX_SET_PRIM_PIVOT_VTX(draws[0].start);
   value |= TGR3D_IDX_SET_PRIM_INVALIDATE_DMACACHE;
   value |= TGR3D_IDX_SET_PRIM_INVALIDATE_VTXCACHE;

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_IDX_SET_PRIM, 1));
   grate_stream_push(stream, value);

   unsigned count = draws[0].count;
   assert(count > 0 && count < (1 << 11));
   value  = TGR3D_IDX_DRAW_PRIM_VTX_COUNT(count - 1);
   value |= TGR3D_IDX_DRAW_PRIM_START_VTX(offset);
   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_IDX_DRAW_PRIM, 1));
   grate_stream_push(stream, value);

   grate_stream_end(stream);

   grate_stream_flush(stream);
}

void
grate_context_draw_init(struct pipe_context *pcontext)
{
   pcontext->draw_vbo = grate_draw_vbo;
}
