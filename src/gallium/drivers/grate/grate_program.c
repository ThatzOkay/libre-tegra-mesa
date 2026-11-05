#include <stdio.h>
#include <string.h>

#include "util/u_dynarray.h"
#include "util/u_memory.h"

#include "nir/nir_to_tgsi.h"

#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "tgsi/tgsi_transform.h"

#include "grate_device.h"
#include "host1x01_hardware.h"
#include "grate_common.h"
#include "grate_context.h"
#include "grate_screen.h"
#include "grate_program.h"
#include "grate_compiler.h"
#include "fp/fpir.h"
#include "vp/vpir.h"

struct grate_vs_tgsi_transform_context {
   struct tgsi_transform_context base;
   unsigned next_temp;
};

static void
grate_vs_tgsi_transform_instruction(struct tgsi_transform_context *ctx,
                                    struct tgsi_full_instruction *inst)
{
   struct grate_vs_tgsi_transform_context *gctx =
      (struct grate_vs_tgsi_transform_context *)ctx;

   int const_index = -1;
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      if (inst->Src[i].Register.File != TGSI_FILE_CONSTANT)
         continue;

      if (const_index < 0)
         const_index = inst->Src[i].Register.Index;
      else if (inst->Src[i].Register.Index != const_index) {
         /* We already have a constant-reads in this instruction,
          * add extra instructions in order to read more
          */
         unsigned temp = gctx->next_temp++;
         tgsi_transform_temp_decl(ctx, temp);
         /* insert MOV to temp */
         tgsi_transform_op1_inst(ctx, TGSI_OPCODE_MOV,
                                 /* dst reg */
                                 TGSI_FILE_TEMPORARY, temp,
                                 TGSI_WRITEMASK_XYZW,
                                 /* src reg */
                                 TGSI_FILE_CONSTANT,
                                 inst->Src[i].Register.Index);

         /* rewrite instruction */
         tgsi_transform_src_reg_xyzw(&inst->Src[i], TGSI_FILE_TEMPORARY, temp);
         inst->Src[i].Register.File = TGSI_FILE_TEMPORARY;
         inst->Src[i].Register.Index = temp;
      }
   }

   ctx->emit_instruction(ctx, inst);
}

static struct tgsi_token *
grate_vs_tgsi_transform(const struct tgsi_token *tokens_in)
{
   grate_trace();
   const unsigned new_len = tgsi_num_tokens(tokens_in) + 100;

   struct tgsi_shader_info info;
   tgsi_scan_shader(tokens_in, &info);

   struct grate_vs_tgsi_transform_context ctx = {};
   ctx.base.transform_instruction = grate_vs_tgsi_transform_instruction;
   ctx.next_temp = info.file_max[TGSI_FILE_TEMPORARY] + 1;

   return tgsi_transform_shader(tokens_in, new_len, &ctx.base);
}


static void *
grate_create_vs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   grate_trace();
   struct grate_vertex_shader_state *so =
      CALLOC_STRUCT(grate_vertex_shader_state);

   if (!so)
      return NULL;

   so->base = *template;

   if (template->type == PIPE_SHADER_IR_NIR) {
      so->base.tokens = nir_to_tgsi(template->ir.nir,
                                    pcontext->screen);
      so->base.type = PIPE_SHADER_IR_TGSI;
   }


   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(so->base.tokens, 0);
      fprintf(stderr, "\n");
   }

   struct tgsi_token *new_tokens = grate_vs_tgsi_transform(so->base.tokens);
   if (!new_tokens)
      return NULL;

   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, new_tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_vp_shader vp;
   grate_tgsi_to_vp(&vp, &parser);

   int num_instructions = list_length(&vp.instructions);
   assert(num_instructions < 256);
   int num_commands = 2 + num_instructions * 4;
   uint32_t *commands = MALLOC(num_commands * sizeof(uint32_t));
   if (!commands) {
      FREE(so);
      return NULL;
   }

   commands[0] = host1x_opcode_imm(REG_TGR3D_VPE_INST_OFFSET, 0);
   commands[1] = host1x_opcode_nonincr(REG_TGR3D_VPE_INST_DATA,
                                       num_instructions * 4);

   struct vp_instr *last = list_last_entry(&vp.instructions, struct vp_instr, link);
   int offset = 2;
   list_for_each_entry(struct vp_instr, instr, &vp.instructions, link) {
      bool end_of_program = instr == last;
      grate_vp_pack(commands + offset, instr, end_of_program);
      offset += 4;
   }

   so->blob.commands = commands;
   so->blob.num_commands = num_commands;
   so->output_mask = vp.output_mask;
   memcpy(so->immediates, vp.immediates, sizeof(so->immediates));
   so->num_immediates = vp.num_immediates;

   return so;
}

static void
grate_bind_vs_state(struct pipe_context *pcontext, void *so)
{
   grate_trace();
   grate_context(pcontext)->vshader = so;
}

static void
grate_delete_vs_state(struct pipe_context *pcontext, void *so)
{
   grate_trace();
   FREE(so);
}

/* Keep the application shader intact and append a source-over stage.
 * Its packed source is R2/R3; PSEQ reads colour target 1 into R0/R1.
 */
static bool
create_premultiplied_blob(const struct grate_shader_blob *source,
                          unsigned fp_count, unsigned alu_count,
                          struct grate_shader_blob *output)
{
   if (fp_count >= 63 || alu_count >= 64)
      return false;
   uint32_t blend_words[8];
   for (unsigned c = 0; c < 4; c++) {
      struct fp_alu_instr instruction = {
         .op = FP_ALU_OP_MAD,
         .dst = {.index = 2 + c / 2, .write_low_sub_reg = !(c & 1),
                 .write_high_sub_reg = c & 1, .saturate = true},
         .src = {
            {.index = c / 2, .datatype = FP_DATATYPE_FIXED10,
             .sub_reg_select_high = c & 1},
            {.index = 3, .datatype = FP_DATATYPE_FIXED10,
             .sub_reg_select_high = true, .minus_one = true, .negate = true},
            {.index = 2 + c / 2, .datatype = FP_DATATYPE_FIXED10,
             .sub_reg_select_high = c & 1},
            {.index = 31, .datatype = FP_DATATYPE_FIXED10, .sub_reg_select_high = true},
         },
      };
      grate_fp_pack_alu(blend_words + c * 2, &instruction);
   }
   struct util_dynarray result;
   util_dynarray_init(&result, NULL);
#define BLEND_PUSH(value) util_dynarray_append_typed(&result, uint32_t, (value))
   unsigned seen = 0;
   for (unsigned i = 0; i < (unsigned)source->num_commands;) {
      uint32_t command = source->commands[i++];
      unsigned opcode = command >> 28;
      unsigned reg = (command >> 16) & 0xfff;
      unsigned count = command & 0xffff;
      if (opcode == 4) {
         if (reg != REG_TGR3D_PSEQ_FLUSH) BLEND_PUSH(command);
         continue;
      }
      if ((opcode != 1 && opcode != 2) || count > (unsigned)source->num_commands - i)
         goto fail;
      const uint32_t *values = source->commands + i;
      i += count;
      if (opcode == 1 && reg == REG_TGR3D_PSEQ_COMMAND_EVEN(0) && count == 1) {
         BLEND_PUSH(command); BLEND_PUSH(0x20006000 | (fp_count + 1));
         continue;
      }
      if (opcode == 1 && reg == REG_TGR3D_PSEQ_DWR_IF_STATE && count == 1) {
         BLEND_PUSH(command); BLEND_PUSH(TGR3D_PSEQ_DWR_IF_STATE_COUNT(fp_count + 1));
         continue;
      }
      unsigned extra = 0, last = 0, bit = 0;
      if (opcode == 2) {
         switch (reg) {
         case REG_TGR3D_PSEQ_INST_DATA: extra = 1; last = 0x00810008; bit = 1; break;
         case REG_TGR3D_AT_REMAP_DATA: extra = 1; bit = 2; break;
         case REG_TGR3D_TEX_INST_DATA: extra = 1; bit = 4; break;
         case REG_TGR3D_ALU_REMAP_DATA: extra = 1; last = (alu_count << 2) | 1; bit = 8; break;
         case REG_TGR3D_ALU_INST_DATA: extra = 8; bit = 16; break;
         case REG_TGR3D_ALU_P2CX_DATA: extra = 1; bit = 32; break;
         case REG_TGR3D_DW_INST_DATA: extra = 1; last = 0x00028005; bit = 64; break;
         }
      }
      if (bit) {
         if ((seen & bit) || count != (bit == 16 ? alu_count * 8 : fp_count)) goto fail;
         seen |= bit;
      }
      BLEND_PUSH((command & 0xffff0000) | (count + extra));
      for (unsigned j = 0; j < count; j++) BLEND_PUSH(bit == 64 ? 0 : values[j]);
      for (unsigned j = 0; j < extra; j++) BLEND_PUSH(bit == 16 ? blend_words[j] : last);
      if (bit == 1) BLEND_PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_FLUSH, 0));
   }
   if (seen != 127) goto fail;
   util_dynarray_trim(&result);
   output->commands = result.data;
   output->num_commands = result.size / sizeof(uint32_t);
#undef BLEND_PUSH
   return true;
fail:
   util_dynarray_fini(&result);
   return false;
}

static void
free_fragment_ir(struct grate_fp_shader *fp)
{
   while (!list_is_empty(&fp->fp_instructions)) {
      struct fp_instr *inst = list_first_entry(&fp->fp_instructions, struct fp_instr, link);
      list_del(&inst->link);
      FREE(inst);
   }
   while (!list_is_empty(&fp->mfu_instructions)) {
      struct fp_mfu_instr *inst = list_first_entry(&fp->mfu_instructions, struct fp_mfu_instr, link);
      list_del(&inst->link);
      FREE(inst);
   }
   while (!list_is_empty(&fp->alu_instructions)) {
      struct fp_alu_instr_packet *inst = list_first_entry(&fp->alu_instructions, struct fp_alu_instr_packet, link);
      list_del(&inst->link);
      FREE(inst);
   }
}

static void *
grate_create_fs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   grate_trace();
   struct grate_context *context = grate_context(pcontext);
   struct grate_fragment_shader_state *so = CALLOC_STRUCT(grate_fragment_shader_state);
   enum drm_tegra_soc_id soc_id = context->drm->soc_id;

   if (!so)
      return NULL;

   so->base = *template;

   if (template->type == PIPE_SHADER_IR_NIR) {
      so->base.tokens = nir_to_tgsi(template->ir.nir,
                                    pcontext->screen);
      so->base.type = PIPE_SHADER_IR_TGSI;
   }


   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(so->base.tokens, 0);
      fprintf(stderr, "\n");
   }

   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, so->base.tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_fp_shader fp;
   grate_tgsi_to_fp(&fp, &parser);
   tgsi_parse_free(&parser);
   if (fp.allocation_failed) {
      free_fragment_ir(&fp);
      if (template->type == PIPE_SHADER_IR_NIR)
         FREE((void *)so->base.tokens);
      FREE(so);
      return NULL;
   }

   struct util_dynarray buf;
   util_dynarray_init(&buf, NULL);

#define PUSH(x) util_dynarray_append_typed(&buf, uint32_t, (x))
   PUSH(host1x_opcode_incr(REG_TGR3D_GLOBAL_PIX_ATTR, 1));
   /* One pixel row, with the matching sequencer capacity used by libgrate.
    * The old capacity/flow-control combination only worked reliably for a
    * single pipeline batch. MUL with independently interpolated operands
    * needs more than one batch.
    */
   PUSH(TGR3D_GLOBAL_PIX_ATTR_MAX_QID((0x12c - 1) / 4));
   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_CTL, 1));
   /* Only R0-R3 keep a pixel's value while many pixels are in flight. R4
    * and up and the globals are storage of four lanes (x mod 4), shared by
    * the pixels in flight, so a program that keeps a value in them needs
    * the sequencer to hold at most 4 pixels (measured: 8 makes neighbouring
    * pixels share a value, see docs/grate-fragment-registers.md). That is
    * several times slower, so only such programs pay for it. */
   PUSH(0x22000000 | TGR3D_PSEQ_CTL_MAX_OUT(fp.uses_globals ? 0x4 : 0x12c) |
        TGR3D_PSEQ_CTL_MIN_OUT(fp.uses_globals ? 0x2 : 0xc8) |
        TGR3D_PSEQ_CTL_MERGE_SPAN_STARTS | TGR3D_PSEQ_CTL_MERGE_REGISTERS |
        TGR3D_PSEQ_CTL_REMOVE_KILLED_PIXELS | TGR3D_PSEQ_CTL_ALLOW_QID_COLLISIONS);

   PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_QUAD_ID, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_GLOBAL_INST_OFFSET, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_AT_INST_OFFSET, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_ALU_INST_OFFSET, 0));

   int num_fp_instrs = list_length(&fp.fp_instructions);
   assert(num_fp_instrs < 64);

   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_COMMAND_EVEN(0), 1));
   // TODO: document/convert these values
   PUSH(0x20006000 | num_fp_instrs);

   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_DWR_IF_STATE, 1));
   // TODO: document/convert these values
   PUSH(TGR3D_PSEQ_DWR_IF_STATE_COUNT(num_fp_instrs));

   if (soc_id == DRM_TEGRA_SOC_T114) {
      // TODO: document/convert these values
      /* XXX: maybe not needed */
      PUSH(host1x_opcode_incr(0x547, 0x0002));
      PUSH(0xc0000000);
      PUSH(0x00000000);
   }

   PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_FLUSH, 0));

   PUSH(host1x_opcode_nonincr(REG_TGR3D_PSEQ_INST_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(0x00000000);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_AT_REMAP_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_sched(&instr->mfu_sched));

   int num_mfu_instrs = list_length(&fp.mfu_instructions);
   assert(num_mfu_instrs < 64); // TODO: not sure if this is really correct

   PUSH(host1x_opcode_nonincr(REG_TGR3D_AT_INST_DATA_LO, num_mfu_instrs * 2));
   list_for_each_entry(struct fp_mfu_instr, instr, &fp.mfu_instructions, link) {
      uint32_t words[2];
      grate_fp_pack_mfu(words, instr);
      PUSH(words[0]);
      PUSH(words[1]);
   }

   PUSH(host1x_opcode_nonincr(REG_TGR3D_TEX_INST_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(instr->tex);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_REMAP_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link) {
      if (soc_id == DRM_TEGRA_SOC_T114)
         PUSH(grate_fp_pack_alu_sched_t114(&instr->alu_sched));
      else
         PUSH(grate_fp_pack_sched(&instr->alu_sched));
   }

   int num_alu_instrs = list_length(&fp.alu_instructions);
   PUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_INST_DATA, num_alu_instrs * 4 * 2));
   list_for_each_entry(struct fp_alu_instr_packet, instr, &fp.alu_instructions, link) {
      for (int i = 0; i < 4; ++i) {
         uint32_t words[2];
         grate_fp_pack_alu(words, instr->slots + i);
         PUSH(words[0]);
         PUSH(words[1]);
      }
   }

   PUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_P2CX_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(0x00000000);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_DW_INST_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_dw(&instr->dw));

   uint32_t tram_setup = 0;
   tram_setup |= TGR3D_GLOBAL_TRI_ATTR_NUM_TRIS(64 / fp.info.max_tram_row);
   tram_setup |= TGR3D_GLOBAL_TRI_ATTR_TRI_ROWS(fp.info.max_tram_row);

   PUSH(host1x_opcode_incr(REG_TGR3D_GLOBAL_TRI_ATTR, 1));
   PUSH(tram_setup);

#undef PUSH
   util_dynarray_trim(&buf);

   so->blob.num_commands = buf.size / sizeof(uint32_t);
   so->blob.commands = buf.data;
   so->info = fp.info;
   free_fragment_ir(&fp);
   if (!create_premultiplied_blob(&so->blob, num_fp_instrs, num_alu_instrs,
                                 &so->premultiplied_blob)) {
      grate_loge("failed to build premultiplied fragment variant\n");
      FREE(so->blob.commands);
      FREE(so);
      return NULL;
   }
   return so;
}

static void
grate_bind_fs_state(struct pipe_context *pcontext, void *so)
{
   grate_trace();
   grate_context(pcontext)->fshader = so;
}

static void
grate_delete_fs_state(struct pipe_context *pcontext, void *so)
{
   struct grate_fragment_shader_state *shader = so;
   grate_trace();
   FREE(shader->blob.commands);
   FREE(shader->premultiplied_blob.commands);
   FREE(shader);
}

void
grate_context_program_init(struct pipe_context *pcontext)
{
   grate_trace();
   pcontext->create_vs_state = grate_create_vs_state;
   pcontext->bind_vs_state = grate_bind_vs_state;
   pcontext->delete_vs_state = grate_delete_vs_state;

   pcontext->create_fs_state = grate_create_fs_state;
   pcontext->bind_fs_state = grate_bind_fs_state;
   pcontext->delete_fs_state = grate_delete_fs_state;
}
