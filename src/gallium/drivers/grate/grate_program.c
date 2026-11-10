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
#include "grate_bo.h"
#include "host1x01_hardware.h"
#include "grate_common.h"
#include "grate_context.h"
#include "grate_screen.h"
#include "grate_program.h"
#include "grate_compiler.h"
#include "fp/blend.h"
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
   tgsi_parse_free(&parser);
   if (vp.failed) {
      /* refused: the draws that need this shader are skipped, see
       * grate_draw_vbo() */
      FREE(so);
      return NULL;
   }

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

/**
 * Keep the application shader intact and append the blend stage for key
 * (see fp/blend.c). The stage is one more sequencer instruction, which
 * reads the destination pixel into R0/R1, and its ALU packets. With more
 * than one register row the program is a whole number of rounds long and the
 * rows of a step are those of its index modulo `rows`: the stage becomes the
 * first step of one more round (it reads the destination into the row 0
 * registers and the output write on it reads R2/R3 of row 0), followed by
 * empty steps to the end of that round.
 */
static bool
create_blend_blob(const struct grate_shader_blob *source, unsigned fp_count,
                  unsigned alu_count, unsigned rows, uint32_t key,
                  struct grate_shader_blob *output)
{
   uint32_t blend_words[GRATE_FP_BLEND_MAX_PACKETS * 8];
   unsigned packets = grate_fp_blend_alu(key, blend_words);

   if (!packets || fp_count + rows > 64 || alu_count + packets > 64 ||
       fp_count % rows)
      return false;
   struct util_dynarray result;
   util_dynarray_init(&result, NULL);
#define BLEND_PUSH(value)                                                    \
   util_dynarray_append_typed(&result, uint32_t, (value))
   unsigned seen = 0;
   for (unsigned i = 0; i < (unsigned)source->num_commands;) {
      uint32_t command = source->commands[i++];
      unsigned opcode = command >> 28;
      unsigned reg = (command >> 16) & 0xfff;
      unsigned count = command & 0xffff;
      if (opcode == 4) {
         if (reg != REG_TGR3D_PSEQ_FLUSH)
            BLEND_PUSH(command);
         continue;
      }
      if ((opcode != 1 && opcode != 2) ||
          count > (unsigned)source->num_commands - i)
         goto fail;
      const uint32_t *values = source->commands + i;
      i += count;
      if (opcode == 1 && reg == REG_TGR3D_PSEQ_COMMAND_EVEN(0) &&
          count == 1) {
         BLEND_PUSH(command);
         BLEND_PUSH(0x20006000 | (fp_count + rows));
         continue;
      }
      if (opcode == 1 && reg == REG_TGR3D_PSEQ_DWR_IF_STATE && count == 1) {
         BLEND_PUSH(command);
         BLEND_PUSH(TGR3D_PSEQ_DWR_IF_STATE_COUNT(fp_count + rows));
         continue;
      }
      unsigned extra = 0, last = 0, bit = 0;
      if (opcode == 2) {
         switch (reg) {
         case REG_TGR3D_PSEQ_INST_DATA:
            extra = rows;
            last = 0x00810008;
            bit = 1;
            break;
         case REG_TGR3D_AT_REMAP_DATA:
            extra = rows;
            bit = 2;
            break;
         case REG_TGR3D_TEX_INST_DATA:
            extra = rows;
            bit = 4;
            break;
         case REG_TGR3D_ALU_REMAP_DATA:
            extra = rows;
            last = (alu_count << 2) | packets;
            bit = 8;
            break;
         case REG_TGR3D_ALU_INST_DATA:
            extra = 8 * packets;
            bit = 16;
            break;
         case REG_TGR3D_ALU_P2CX_DATA:
            extra = rows;
            bit = 32;
            break;
         case REG_TGR3D_DW_INST_DATA:
            extra = rows;
            last = 0x00028005;
            bit = 64;
            break;
         }
      }
      if (bit) {
         if ((seen & bit) || count != (bit == 16 ? alu_count * 8 : fp_count))
            goto fail;
         seen |= bit;
      }
      BLEND_PUSH((command & 0xffff0000) | (count + extra));
      for (unsigned j = 0; j < count; j++)
         BLEND_PUSH(
            bit == 64 && (values[j] & 3) == TGR3D_DATA_OP_ST ? 0 : values[j]);
      for (unsigned j = 0; j < extra; j++) {
         if (bit == 16)
            BLEND_PUSH(blend_words[j]);
         else
            BLEND_PUSH(j == 0 ? last : 0);
      }
      if (bit == 1)
         BLEND_PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_FLUSH, 0));
   }
   if (seen != 127)
      goto fail;
   util_dynarray_trim(&result);
   output->commands = result.data;
   output->num_commands = result.size / sizeof(uint32_t);
#undef BLEND_PUSH
   return true;
fail:
   util_dynarray_fini(&result);
   return false;
}

/**
 * The fragment program with the blend stage for key, built on first use.
 * Returns NULL if it cannot be built.
 */
const struct grate_shader_blob *
grate_fragment_blend_blob(struct grate_fragment_shader_state *shader,
                          uint32_t key)
{
   struct grate_blend_variant *variant;

   for (variant = shader->variants; variant; variant = variant->next) {
      if (variant->key == key)
         return &variant->blob;
   }

   variant = CALLOC_STRUCT(grate_blend_variant);
   if (!variant)
      return NULL;

   variant->key = key;
   if (!create_blend_blob(&shader->blob, shader->num_fp_instrs,
                          shader->num_alu_instrs, shader->info.num_rows, key,
                          &variant->blob)) {
      grate_loge("cannot build the blend variant for key 0x%08x "
                 "(fragment program too long for the blend stage?)\n",
                 key);
      FREE(variant);
      return NULL;
   }

   variant->next = shader->variants;
   shader->variants = variant;
   return &variant->blob;
}

static void
free_fragment_ir(struct grate_fp_shader *fp)
{
   while (!list_is_empty(&fp->fp_instructions)) {
      struct fp_instr *inst =
         list_first_entry(&fp->fp_instructions, struct fp_instr, link);
      list_del(&inst->link);
      FREE(inst);
   }
   while (!list_is_empty(&fp->mfu_instructions)) {
      struct fp_mfu_instr *inst =
         list_first_entry(&fp->mfu_instructions, struct fp_mfu_instr, link);
      list_del(&inst->link);
      FREE(inst);
   }
   while (!list_is_empty(&fp->alu_instructions)) {
      struct fp_alu_instr_packet *inst = list_first_entry(
         &fp->alu_instructions, struct fp_alu_instr_packet, link);
      list_del(&inst->link);
      FREE(inst);
   }
}

static bool
scalar_packet_masks(const struct fp_alu_instr_packet *packet, uint8_t *reads,
                    uint8_t *writes, unsigned *active)
{
   memset(reads, 0, 72);
   memset(writes, 0, 72);
   *active = 0;
   for (unsigned slot = 0; slot < 4; slot++) {
      const struct fp_alu_instr *alu = &packet->slots[slot];
      unsigned mask = alu->dst.write_low_sub_reg |
                      ((unsigned)alu->dst.write_high_sub_reg << 1);
      if (!mask)
         continue;
      unsigned reg = alu->dst.index;
      if (!(reg < 16 || (reg >= 64 && reg < 72)))
         return false;
      writes[reg] |= mask;
      for (unsigned s = 0; s < 4; s++) {
         const struct fp_alu_src_operand *src = &alu->src[s];
         reg = src->index;
         if (reg < 16 || (reg >= 64 && reg < 72)) {
            reads[reg] |= src->datatype == FP_DATATYPE_FIXED10
                             ? 1u << src->sub_reg_select_high
                             : 3;
         } else if (reg < 31 || reg >= 64) {
            /* In particular, R24-R27 forward packet results. */
            return false;
         }
      }
      (*active)++;
   }
   return *active != 0;
}

static bool
plain_scalar_step(const struct fp_instr *instr)
{
   return instr->alu_sched.num_instructions == 1 &&
          !instr->mfu_sched.num_instructions && !instr->tex &&
          !instr->dw.enable && !instr->dw.spill && !instr->pseq &&
          !instr->spill_load;
}

/* Scalarizing the spill retry exposes independent lanes as separate
 * steps. Pack them back together when all slots can read the old register
 * values: a later packet must not read or overwrite an earlier result. */
static void
coalesce_scalar_alu(struct grate_fp_shader *fp)
{
   if (list_is_empty(&fp->fp_instructions))
      return;
   unsigned num_packets = list_length(&fp->alu_instructions);
   struct fp_alu_instr_packet **packets =
      CALLOC(num_packets, sizeof(*packets));
   if (!packets)
      return;
   unsigned index = 0;
   list_for_each_entry (struct fp_alu_instr_packet, packet,
                        &fp->alu_instructions, link)
      packets[index++] = packet;

   struct fp_instr *instr =
      list_first_entry(&fp->fp_instructions, struct fp_instr, link);
   while (instr->link.next != &fp->fp_instructions) {
      struct fp_instr *next =
         list_entry(instr->link.next, struct fp_instr, link);
      uint8_t reads[72], writes[72], next_reads[72], next_writes[72];
      unsigned active, next_active;
      bool merge = plain_scalar_step(instr) && plain_scalar_step(next) &&
                   instr->alu_sched.address < num_packets &&
                   next->alu_sched.address < num_packets &&
                   instr->alu_sched.address != next->alu_sched.address;
      if (merge) {
         merge = scalar_packet_masks(packets[instr->alu_sched.address], reads,
                                     writes, &active) &&
                 scalar_packet_masks(packets[next->alu_sched.address],
                                     next_reads, next_writes, &next_active) &&
                 active + next_active <= 4;
      }
      if (merge) {
         for (unsigned reg = 0; reg < 72; reg++)
            if (writes[reg] & (next_reads[reg] | next_writes[reg]))
               merge = false;
      }
      if (!merge) {
         instr = next;
         continue;
      }
      struct fp_alu_instr_packet *dst = packets[instr->alu_sched.address];
      const struct fp_alu_instr_packet *src =
         packets[next->alu_sched.address];
      unsigned empty = 0;
      for (unsigned slot = 0; slot < 4; slot++) {
         if (!src->slots[slot].dst.write_low_sub_reg &&
             !src->slots[slot].dst.write_high_sub_reg)
            continue;
         while (dst->slots[empty].dst.write_low_sub_reg ||
                dst->slots[empty].dst.write_high_sub_reg)
            empty++;
         dst->slots[empty++] = src->slots[slot];
      }
      list_del(&next->link);
      FREE(next);
   }
   FREE(packets);
}

static bool
step_writes_register(const struct fp_instr *instr,
                     struct fp_alu_instr_packet **packets,
                     unsigned num_packets, unsigned reg)
{
   for (unsigned p = 0; p < instr->alu_sched.num_instructions; p++) {
      unsigned address = instr->alu_sched.address + p;
      if (address >= num_packets)
         return true;
      for (unsigned slot = 0; slot < 4; slot++) {
         const struct fp_alu_dst_operand *dst =
            &packets[address]->slots[slot].dst;
         if (dst->index == reg &&
             (dst->write_low_sub_reg || dst->write_high_sub_reg))
            return true;
      }
   }
   return false;
}

static void
append_row_step(struct list_head *steps, struct fp_instr *instr,
                unsigned *count, unsigned rows, uint32_t *pending_pseq,
                bool *pending_store)
{
   list_addtail(&instr->link, steps);
   (*count)++;
   if (*count % rows == 0) {
      if (*pending_pseq)
         instr->pseq = *pending_pseq;
      *pending_pseq = 0;
      *pending_store = false;
   }
}

/**
 * Lay the steps out for a program with more than one register row.
 *
 * The ALU packets name absolute registers, but the units around them work
 * on the row `step % rows` (R0-R3 of that row are R(4k)..R(4k+3), k =
 * step % rows): the interpolator writes it, the texture unit reads its
 * coordinates from it and writes its result to R2/R3 of it, and the output
 * write reads the colour from R0/R1 or R2/R3 of the row of the step that
 * holds it. NVIDIA's programs also always start with ALU work, the program
 * is a whole number of rounds long, and (measured on the tablet) these are
 * all needed. So: put every step with an interpolation, an SFU op or a
 * fetch on a multiple of `rows` (the row that our ALU code names R0-R3),
 * with empty steps in between, pad the program to a multiple of `rows`,
 * put the output write on the first step of the last round (it fires at the
 * end of its round and reads R2/R3 of row 0 there) and give every round an
 * ALU packet on its first step, which writes nothing if there is no other.
 */
static void
align_to_rows(struct grate_fp_shader *fp)
{
   unsigned rows = fp->info.num_rows;

   if (rows <= 1 || list_is_empty(&fp->fp_instructions))
      return;

   unsigned num_packets = list_length(&fp->alu_instructions);
   struct fp_alu_instr_packet **old = CALLOC(num_packets + 1, sizeof(*old));
   if (!old) {
      fp->allocation_failed = true;
      return;
   }
   unsigned num_old = 0;
   list_for_each_entry (struct fp_alu_instr_packet, packet,
                        &fp->alu_instructions, link)
      old[num_old++] = packet;

   struct fp_dw_instr dw = {0};
   list_for_each_entry(struct fp_instr, instr, &fp->fp_instructions, link) {
      if (instr->dw.enable && !instr->dw.spill) {
         dw = instr->dw;
         memset(&instr->dw, 0, sizeof(instr->dw));
      }
   }

   struct list_head steps;
   list_inithead(&steps);
   unsigned count = 0;
   uint32_t pending_pseq = 0;
   bool pending_store = false;
   list_for_each_entry_safe(struct fp_instr, instr, &fp->fp_instructions,
                            link) {
      bool row_unit = instr->mfu_sched.num_instructions ||
                      (instr->tex & (1 << 10));
      bool memory = instr->dw.spill || instr->spill_load;
      bool pending = pending_store || pending_pseq;
      bool protected_write =
         (pending_store && step_writes_register(instr, old, num_old, 0)) ||
         (pending_pseq && step_writes_register(instr, old, num_old, 12));

      list_del(&instr->link);
      while (count % rows && (memory || protected_write ||
                              (pending && instr->pseq) || row_unit)) {
         struct fp_instr *spacer = CALLOC_STRUCT(fp_instr);
         list_inithead(&spacer->link);
         append_row_step(&steps, spacer, &count, rows, &pending_pseq,
                         &pending_store);
      }
      /* Store R0 at round end; a row-3 load supplies R12 to the row-0
       * consumer. Other ALU work can fill the round while preserving
       * those registers and leaving the other row units idle. */
      if (instr->dw.spill)
         pending_store = true;
      if (instr->spill_load) {
         pending_pseq = instr->pseq;
         instr->pseq = 0;
      }
      append_row_step(&steps, instr, &count, rows, &pending_pseq,
                      &pending_store);
   }
   while (count % rows) {
      struct fp_instr *spacer = CALLOC_STRUCT(fp_instr);
      list_inithead(&spacer->link);
      append_row_step(&steps, spacer, &count, rows, &pending_pseq,
                      &pending_store);
   }
   list_replace(&steps, &fp->fp_instructions);

   /* An output store must not overwrite a spill store in the last round. */
   if (dw.enable) {
      unsigned index = 0;
      bool last_spill_store = false;
      list_for_each_entry (struct fp_instr, instr, &fp->fp_instructions,
                           link) {
         if (index == count - rows)
            last_spill_store = instr->dw.spill;
         index++;
      }
      if (last_spill_store) {
         for (unsigned row = 0; row < rows; row++) {
            struct fp_instr *spacer = CALLOC_STRUCT(fp_instr);
            list_inithead(&spacer->link);
            list_addtail(&spacer->link, &fp->fp_instructions);
            count++;
         }
      }
   }

   /* The output write fires at the end of the round of the step it is on and
    * reads R2/R3 of the row `step % rows`: the first step of the last round
    * sees all the ALU work and reads row 0. */
   unsigned index = 0;
   list_for_each_entry(struct fp_instr, instr, &fp->fp_instructions, link) {
      if (dw.enable && index == count - rows)
         instr->dw = dw;
      index++;
   }

   /* Rebuild the list of ALU packets, one run per step:
    * - an ALU packet on a step that is not the first of its round runs only
    *   when the first step of the round has an ALU packet too, so a round
    *   that has none gets a packet that writes nothing;
    * - a dependent ALU packet needs a packet in between (two packet
    *   latency): between two steps of the same round there is none, so a
    *   step whose successor in the round has ALU work ends with a packet
    *   that writes nothing (a step change between rounds is enough). */
   struct list_head packets;
   list_inithead(&packets);
   unsigned address = 0, step = 0;
   list_for_each_entry(struct fp_instr, instr, &fp->fp_instructions, link) {
      unsigned first = instr->alu_sched.address;
      unsigned num = instr->alu_sched.num_instructions;
      struct fp_instr *next =
         instr->link.next == &fp->fp_instructions
            ? NULL
            : list_entry(instr->link.next, struct fp_instr, link);
      bool round_start = step % rows == 0;
      bool next_in_round = next && (step + 1) % rows != 0;
      bool next_has_alu = next && (next->alu_sched.num_instructions ||
                                   next->mfu_sched.num_instructions ||
                                   (next->tex & (1 << 10)));
      unsigned nops_before = 0, nops_after = 0;

      if (round_start && num == 0)
         nops_before = 1;
      if (next_in_round && next_has_alu && (num || nops_before))
         nops_after = 1;
      /* the next step is the first of a round: it needs nothing; if the
       * next step is not a round start but has ALU work while this one has
       * none, there is nothing to wait for either */

      instr->alu_sched.address = address;
      instr->alu_sched.num_instructions = num + nops_before + nops_after;
      if (instr->alu_sched.num_instructions > 3) {
         fp->allocation_failed = true;
         fprintf(stderr, "GRATE FRAG: too many ALU packets in step %u of a "
                         "multi-row program\n", step);
      }
      for (unsigned i = 0; i < nops_before + num + nops_after; i++) {
         struct fp_alu_instr_packet *copy =
            CALLOC_STRUCT(fp_alu_instr_packet);
         list_inithead(&copy->link);
         if (i >= nops_before && i < nops_before + num) {
            *copy = *old[first + i - nops_before];
            list_inithead(&copy->link);
         } else {
            for (unsigned slot = 0; slot < 4; slot++)
               for (unsigned operand = 0; operand < 4; operand++)
                  copy->slots[slot].src[operand].index = 31;
         }
         list_addtail(&copy->link, &packets);
         address++;
      }
      step++;
   }
   list_for_each_entry_safe(struct fp_alu_instr_packet, packet,
                            &fp->alu_instructions, link) {
      list_del(&packet->link);
      FREE(packet);
   }
   FREE(old);
   list_replace(&packets, &fp->alu_instructions);
}

/**
 * Compiles the program of so->base.tokens, storing the output the mode asks
 * for, into so->blob and so->info.
 */
static bool
grate_compile_fs(struct grate_context *context,
                 struct grate_fragment_shader_state *so,
                 const struct grate_fp_output *output)
{
   enum drm_tegra_soc_id soc_id = context->drm->soc_id;

   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, so->base.tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_fp_shader fp;
   grate_tgsi_to_fp(&fp, &parser, output, false);
   if (fp.allocation_failed && fp.register_pressure) {
      free_fragment_ir(&fp);
      tgsi_parse_free(&parser);
      ok = tgsi_parse_init(&parser, so->base.tokens);
      if (ok != TGSI_PARSE_OK)
         return false;
      grate_tgsi_to_fp(&fp, &parser, output, true);
   }
   tgsi_parse_free(&parser);
   if (fp.allocation_failed) {
      free_fragment_ir(&fp);
      return false;
   }

   /* Register rows: R0-R3 are row 1, R4-R7 row 2, and so on. The hardware
    * keeps that many rows per pixel in flight, so a program that writes R11
    * needs three, and fewer pixels fit in flight. */
   unsigned max_reg = 0;
   list_for_each_entry(struct fp_alu_instr_packet, packet,
                       &fp.alu_instructions, link) {
      for (unsigned slot = 0; slot < 4; slot++) {
         const struct fp_alu_dst_operand *dst = &packet->slots[slot].dst;
         if ((dst->write_low_sub_reg || dst->write_high_sub_reg) &&
             dst->index < 16)
            max_reg = MAX2(max_reg, dst->index);
      }
   }
   /* The quad exchange needs three register rows, as in NVIDIA's code. */
   if (fp.uses_derivatives)
      max_reg = MAX2(max_reg, 11);
   fp.info.num_rows = max_reg / 4 + 1;

   bool has_spills = false;
   list_for_each_entry (struct fp_instr, instr, &fp.fp_instructions, link)
      has_spills |= instr->dw.spill || instr->spill_load;
   if (has_spills)
      fp.info.num_rows = 4;

   if (fp.spilling)
      coalesce_scalar_alu(&fp);
   align_to_rows(&fp);

   unsigned num_fp_instrs = list_length(&fp.fp_instructions);
   unsigned num_mfu_instrs = list_length(&fp.mfu_instructions);
   if (fp.allocation_failed || num_fp_instrs > 64 || num_mfu_instrs >= 64) {
      if (!fp.allocation_failed)
         fprintf(stderr,
                 "GRATE FRAG: program exceeds one epoch "
                 "(%u steps, %u spill slots)\n",
                 num_fp_instrs, fp.info.num_spill_slots);
      free_fragment_ir(&fp);
      return false;
   }

   if (has_spills) {
      fp.info.num_spill_slots = MAX2(fp.info.num_spill_slots, 1);
      /* Four FP20 words occupy each P128 element. NVIDIA uses a row
       * stride of 16 bytes times four lanes times MAX_QID + 1. */
      unsigned stride = GRATE_FP_SPILL_STRIDE;
      unsigned size = DIV_ROUND_UP(fp.info.num_spill_slots, 4) * stride;
      if (size > context->spill_size) {
         struct grate_bo *bo = grate_bo_alloc(context->drm, size, 0);
         if (!bo) {
            free_fragment_ir(&fp);
            return false;
         }
         grate_bo_unref(context->spill_bo);
         context->spill_bo = bo;
         context->spill_size = size;
      }
   }

   struct util_dynarray buf;
   util_dynarray_init(&buf, NULL);

#define PUSH(x) util_dynarray_append_typed(&buf, uint32_t, (x))
   PUSH(host1x_opcode_incr(REG_TGR3D_GLOBAL_PIX_ATTR, 1));
   /* The register file holds `capacity` pixel rows in flight: a program
    * with more register rows (R4-R7, R8-R11, ...) gets proportionally fewer
    * quad ids. The numbers are those of NVIDIA's driver for a capacity of
    * 0x12c (NvArdLoadFragmentProgram_SingleEpoch_General: MAX_QID =
    * (capacity - 1) >> 2, >> 3, / 12, >> 4 for 1 to 4 rows). A value in R4
    * or above is only kept per pixel when NUM_ROWS covers it; the
    * programs that used the globals 16-23 for it needed MAX_OUT 4. */
   assert(fp.info.num_rows >= 1 && fp.info.num_rows <= 4);
   PUSH(TGR3D_GLOBAL_PIX_ATTR_MAX_QID(grate_fp_max_qid(fp.info.num_rows)) |
        TGR3D_GLOBAL_PIX_ATTR_NUM_ROWS(fp.info.num_rows - 1));
   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_CTL, 1));
   PUSH(0x22000000 | TGR3D_PSEQ_CTL_MAX_OUT(0x12c) |
        TGR3D_PSEQ_CTL_MIN_OUT((has_spills || fp.info.epochs) ? 0 : 0xc8) |
        TGR3D_PSEQ_CTL_MERGE_SPAN_STARTS | TGR3D_PSEQ_CTL_MERGE_REGISTERS |
        TGR3D_PSEQ_CTL_REMOVE_KILLED_PIXELS |
        (has_spills ? 0 : TGR3D_PSEQ_CTL_ALLOW_QID_COLLISIONS));

   PUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_QUAD_ID, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_GLOBAL_INST_OFFSET, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_AT_INST_OFFSET, 0));
   PUSH(host1x_opcode_imm(REG_TGR3D_ALU_INST_OFFSET, 0));

   size_t cmd_start = buf.size;
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
      PUSH(instr->pseq);

   PUSH(host1x_opcode_nonincr(REG_TGR3D_AT_REMAP_DATA, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_sched(&instr->mfu_sched));

   PUSH(host1x_opcode_nonincr(REG_TGR3D_AT_INST_DATA_LO, num_mfu_instrs * 2));
   list_for_each_entry(struct fp_mfu_instr, instr, &fp.mfu_instructions, link) {
      uint32_t words[2];
      grate_fp_pack_mfu(words, instr);
      PUSH(words[0]);
      PUSH(words[1]);
   }

   PUSH(host1x_opcode_nonincr(REG_TGR3D_TEX_INST_DATA, num_fp_instrs));
   list_for_each_entry (struct fp_instr, instr, &fp.fp_instructions, link)
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

   /* A program of more than one epoch is loaded by the sequencer's DMA,
    * from the buffers of grate_fp_epochs: keep the common state only. */
   if (fp.info.epochs)
      buf.size = cmd_start;

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
   so->num_fp_instrs = num_fp_instrs;
   so->num_alu_instrs = num_alu_instrs;
   return true;
}

static void *
grate_create_fs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   grate_trace();
   struct grate_context *context = grate_context(pcontext);
   struct grate_fragment_shader_state *so =
      CALLOC_STRUCT(grate_fragment_shader_state);
   const struct grate_fp_output output = {.mode = GRATE_FP_OUTPUT_SRC0};

   if (!so)
      return NULL;

   so->base = *template;

   if (template->type == PIPE_SHADER_IR_NIR) {
      so->base.tokens = nir_to_tgsi(template->ir.nir, pcontext->screen);
      so->base.type = PIPE_SHADER_IR_TGSI;
   }

   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(so->base.tokens, 0);
      fprintf(stderr, "\n");
   }

   if (!grate_compile_fs(context, so, &output)) {
      if (template->type == PIPE_SHADER_IR_NIR)
         FREE((void *)so->base.tokens);
      FREE(so);
      return NULL;
   }

   return so;
}

/**
 * The program that stores what a dual-source pass blends (see
 * fp/blend.h), compiled on first use. NULL if it cannot be compiled.
 */
struct grate_fragment_shader_state *
grate_fragment_dual_variant(struct grate_context *context,
                            struct grate_fragment_shader_state *shader,
                            const struct grate_fp_output *output)
{
   struct grate_fragment_shader_state *variant;

   for (variant = shader->dual_variants; variant;
        variant = variant->next_dual) {
      if (!memcmp(&variant->output, output, sizeof(*output)))
         return variant;
   }

   variant = CALLOC_STRUCT(grate_fragment_shader_state);
   if (!variant)
      return NULL;

   /* shares the tokens; the variant never frees them */
   variant->base = shader->base;
   variant->output = *output;
   if (!grate_compile_fs(context, variant, output)) {
      grate_loge("cannot compile the dual-source variant of the "
                 "fragment program\n");
      FREE(variant);
      return NULL;
   }

   variant->next_dual = shader->dual_variants;
   shader->dual_variants = variant;
   return variant;
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
   if (shader->info.epochs) {
      grate_bo_unref(shader->info.epochs->program_bo);
      grate_bo_unref(shader->info.epochs->constant_bo);
      FREE(shader->info.epochs->program);
      FREE(shader->info.epochs);
   }
   while (shader->dual_variants) {
      struct grate_fragment_shader_state *variant = shader->dual_variants;

      shader->dual_variants = variant->next_dual;
      grate_delete_fs_state(pcontext, variant);
   }
   while (shader->variants) {
      struct grate_blend_variant *variant = shader->variants;

      shader->variants = variant->next;
      FREE(variant->blob.commands);
      FREE(variant);
   }
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
