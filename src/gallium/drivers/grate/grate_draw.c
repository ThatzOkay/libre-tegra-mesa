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
    * 0x75x should be written after REG_TGR3D_GSHIM_WRITE_MASK, otherwise non-pow2
    * textures are corrupted. Reason is unknown. Looks like
    * combination of REG_TGR3D_TEX_TEXDESC_NPOT_AUX(*) register bits affects the texture size.
    *
    * The REG_TGR3D_TEX_TEXDESC_NPOT_AUX(*) registers contain garbage after machine's power-off,
    * but values are retained on soft reboot.
    */
   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_TEX_TEXDESC_NPOT_AUX(0),
                                                      REG_TGR3D_TEX_TEXDESC_NPOT_AUX_LENGTH));
   for (int i = 0; i < REG_TGR3D_TEX_TEXDESC_NPOT_AUX_LENGTH; i++)
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
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_STALL(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_STALL(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_WAIT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_WAIT(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_COMB(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_COMB(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_HWR_WAIT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_HWR_XFER, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_SYNCPT_WAIT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_STAT_SYNCPT_WAIT(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DLB_CONTROL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DLB_RANGE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DLB_TRIGGER, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GSHIM_DEBUG0, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_MEMORY_OUTPUT_READS, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_HORIZONTAL_SWATH_RENDERING, 0));

   /* Common stuff */
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_CLK_COUNT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_CLK_COUNT(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_XFER_COUNT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_XFER_COUNT(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_WAIT_COUNT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_WAIT_COUNT(1), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_EN_COUNT(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_CTL_STAT_EN_COUNT(1), 0));

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
   grate_stream_push(stream, 0x00000011); // REG_TGR3D_VPE_MODE
   grate_stream_push(stream, 0x0000ffff); // REG_TGR3D_VPE_TIMEOUT
   grate_stream_push(stream, 0x00ff0000); // REG_TGR3D_VPE_CONST_READ_LIMIT
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_VPE_BRANCHBITS
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_VPE_START

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_VPE_GEOM_STALL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_VPE_VPE_CTRL, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_VPE_VPE_DEBUG,
                                               TGR3D_VPE_VPE_DEBUG_VPE_DEBUG_IBUF_SIZE |
                                               TGR3D_VPE_VPE_DEBUG_VPE_DEBUG_OBUF_SIZE));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_INST_EVEN(0), 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_INST_EVEN(1), 0));

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_SU_PARAM, 25));
   // TODO: document/convert these values
   grate_stream_push(stream, 0xb8e00000); // REG_TGR3D_SU_PARAM
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_SU_ZBIAS
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_SU_ZFACTOR
   grate_stream_push(stream, 0x00000105); // REG_TGR3D_SU_POINT_PARAM
   grate_stream_push(stream, u_bitcast_f2u(0.5f)); // REG_TGR3D_SU_POINT_WIDTH_2
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); // REG_TGR3D_SU_POINT_MAX_S
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); // REG_TGR3D_SU_POINT_MAX_T
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); // REG_TGR3D_SU_POINT_MIN_S
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); // REG_TGR3D_SU_POINT_MIN_T
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_SU_LINE_PARAM
   grate_stream_push(stream, u_bitcast_f2u(0.5f)); // REG_TGR3D_SU_LINE_WIDTH_2
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); // REG_TGR3D_SU_LINE_MAX_ATTR_W
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_SU_LINE_MIN_ATTR_W
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_SU_SCISSOR_X
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_SU_SCISSOR_Y
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); // REG_TGR3D_SU_VIEWPORT_X
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); // REG_TGR3D_SU_VIEWPORT_Y
   grate_stream_push(stream, u_bitcast_f2u(0.5f - powf(2.0, -21))); // REG_TGR3D_SU_VIEWPORT_Z
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); // REG_TGR3D_SU_VIEWPORT_W
   grate_stream_push(stream, u_bitcast_f2u(0.0f)); // REG_TGR3D_SU_VIEWPORT_H
   grate_stream_push(stream, u_bitcast_f2u(0.5f - powf(2.0, -21))); // REG_TGR3D_SU_VIEWPORT_D
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); // REG_TGR3D_SU_GUARDBAND_W
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); // REG_TGR3D_SU_GUARDBAND_H
   grate_stream_push(stream, u_bitcast_f2u(1.0f)); // REG_TGR3D_SU_GUARDBAND_D
   grate_stream_push(stream, 0x00000205); // REG_TGR3D_SU_UCPLANE

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_CLKEN_OVERRIDE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_SU_CLIP_CLKEN_OVERRIDE, 0));

   grate_stream_push(stream, host1x_opcode_incr(REG_TGR3D_QR_S_TEST_FRONT, 20));
   grate_stream_push(stream, TGR3D_QR_S_TEST_FRONT_S_MASK(0xff) |
                                   TGR3D_QR_S_TEST_FRONT_S_FUNC(0x07)); // REG_TGR3D_QR_S_TEST_FRONT
   grate_stream_push(stream, TGR3D_QR_S_TEST_BACK_S_MASK(0xff) |
                                   TGR3D_QR_S_TEST_BACK_S_FUNC(0x07)); // REG_TGR3D_QR_S_TEST_BACK
   grate_stream_push(stream, 0x00000040); // REG_TGR3D_QR_S_CTRL
   grate_stream_push(stream, 0x00000310); // REG_TGR3D_QR_Z_TEST
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_Z_MIN
   grate_stream_push(stream, 0x000fffff); // REG_TGR3D_QR_Z_MAX
   grate_stream_push(stream, 0x00000001); // REG_TGR3D_QR_RAST_OPERATION
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_RAST_SCISSOR_SNAP
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_RAST_SCISSOR_MIN
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_RAST_SCISSOR_MAX
   grate_stream_push(stream, 0x1fff1fff); // REG_TGR3D_QR_RAST_BBOX_MIN
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_RAST_BBOX_MAX
   grate_stream_push(stream, 0x00000006); // REG_TGR3D_QR_SB_OPERATION
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_QRAST_CLKEN_OVERRIDE
   grate_stream_push(stream, 0x00000008); // REG_TGR3D_QR_VCAA_OPERATION
   grate_stream_push(stream, 0x00000048); // REG_TGR3D_QR_OUTPUT_TO_SHADER
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_QRAST_DEBUG
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_QRAST_LIMITS
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_PIXEL_COUNT_CTRL
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_QR_PIXEL_COUNT

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
   grate_stream_push(stream, 0x00000e00); // REG_TGR3D_FDC_CONTROL
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_FDC_STATUS
   grate_stream_push(stream, 0x000001ff); // REG_TGR3D_FDC_MAX_QZ_LINES
   grate_stream_push(stream, 0x000001ff); // REG_TGR3D_FDC_MAX_QV_LINES
   grate_stream_push(stream, 0x000001ff); // REG_TGR3D_FDC_MAX_QS_LINES
   grate_stream_push(stream, 0x00000030); // REG_TGR3D_FDC_MAX_PS_LINES
   grate_stream_push(stream, 0x00000020); // REG_TGR3D_FDC_MAX_Q_LINES
   grate_stream_push(stream, 0x000001ff); // REG_TGR3D_FDC_MAX_Q_P_LINES
   grate_stream_push(stream, 0x00000100); // REG_TGR3D_FDC_FLUSH_CTL
   grate_stream_push(stream, 0x0f0f0f0f); // REG_TGR3D_FDC_L1_TIMEOUT
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_FDC_INSTRUMENT
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_FDC_CLKEN_OVERRIDE
   grate_stream_push(stream, 0x00000000); // REG_TGR3D_FDC_NV_MCCIF_FIFOCTRL_RO

   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_PIX_ATTR, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_TRI_ATTR, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_INST_OFFSET, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_INSTRUMENT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_DITHER_TABLE, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_FLUSH, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_S_OPERATION_FRONT, 0));
   grate_stream_push(stream, host1x_opcode_imm(REG_TGR3D_GLOBAL_S_OPERATION_BACK, 0));

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
