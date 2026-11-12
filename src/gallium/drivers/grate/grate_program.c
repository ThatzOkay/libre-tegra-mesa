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

   if (!packets) {
      grate_loge("blend: no ALU packets for key 0x%08x\n", key);
      return false;
   }
   if (fp_count + rows > 64) {
      grate_loge("blend: %u sequencer steps + %u rows > 64\n", fp_count,
                 rows);
      return false;
   }
   if (alu_count + packets > 64) {
      grate_loge("blend: %u ALU packets + %u blend packets > 64\n", alu_count,
                 packets);
      return false;
   }
   if (fp_count % rows) {
      grate_loge("blend: %u steps is not a whole number of %u rows\n",
                 fp_count, rows);
      return false;
   }
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
 * The blend variant of a program of more than one epoch. NVIDIA puts the
 * blend stage in the last epoch (as its last step, with the destination
 * read) and leaves the others alone, so the last block is rewritten like
 * the whole program of one epoch is and the variant gets a copy of the
 * epochs with that block replaced. The common state stays the same.
 */
static bool
create_blend_epochs(const struct grate_fragment_shader_state *shader,
                    uint32_t key, struct grate_blend_variant *variant)
{
   const struct grate_fp_epochs *epochs = shader->info.epochs;
   const struct grate_fp_epoch *last = &epochs->epoch[epochs->count - 1];
   struct grate_shader_blob block = {
      .commands =
         (uint32_t *)((uint8_t *)epochs->program + last->prog_offset),
      .num_commands = last->prog_words,
   };
   struct grate_shader_blob blended;
   uint32_t words[GRATE_FP_BLEND_MAX_PACKETS * 8];
   const unsigned blend_packets = grate_fp_blend_alu(key, words);

   if (!create_blend_blob(&block, last->steps, last->packets,
                          shader->info.num_rows, key, &blended))
      return false;

   struct grate_fp_epochs *copy = MALLOC_STRUCT(grate_fp_epochs);
   if (!copy) {
      FREE(blended.commands);
      return false;
   }
   *copy = *epochs;
   copy->program_bo = NULL;
   copy->constant_bo = NULL;
   copy->program_bytes =
      last->prog_offset + blended.num_commands * sizeof(uint32_t);
   copy->program = MALLOC(copy->program_bytes);
   uint32_t *common = MALLOC(shader->blob.num_commands * sizeof(uint32_t));
   if (!copy->program || !common) {
      FREE(copy->program);
      FREE(common);
      FREE(copy);
      FREE(blended.commands);
      return false;
   }
   memcpy(copy->program, epochs->program, last->prog_offset);
   memcpy((uint8_t *)copy->program + last->prog_offset, blended.commands,
          blended.num_commands * sizeof(uint32_t));
   FREE(blended.commands);
   copy->epoch[epochs->count - 1].prog_words = blended.num_commands;
   copy->epoch[epochs->count - 1].steps += shader->info.num_rows;
   copy->epoch[epochs->count - 1].packets += blend_packets;

   memcpy(common, shader->blob.commands,
          shader->blob.num_commands * sizeof(uint32_t));
   variant->blob.commands = common;
   variant->blob.num_commands = shader->blob.num_commands;
   variant->epochs = copy;
   return true;
}

/**
 * The fragment program with the blend stage for key, built on first use.
 * Returns NULL if it cannot be built.
 */
const struct grate_blend_variant *
grate_fragment_blend_variant(struct grate_fragment_shader_state *shader,
                             uint32_t key)
{
   struct grate_blend_variant *variant;

   for (variant = shader->variants; variant; variant = variant->next) {
      if (variant->key == key)
         return variant;
   }

   variant = CALLOC_STRUCT(grate_blend_variant);
   if (!variant)
      return NULL;

   variant->key = key;
   bool ok;
   if (shader->info.epochs) {
      ok = create_blend_epochs(shader, key, variant);
   } else {
      ok = create_blend_blob(&shader->blob, shader->num_fp_instrs,
                             shader->num_alu_instrs, shader->info.num_rows,
                             key, &variant->blob);
   }
   if (!ok) {
      grate_loge("cannot build the blend variant for key 0x%08x "
                 "(fragment program too long for the blend stage?)\n",
                 key);
      FREE(variant);
      return NULL;
   }

   variant->next = shader->variants;
   shader->variants = variant;
   return variant;
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

/*
 * Read-only diagnostic (GRATE_PACKING_DEBUG): what do the slot operations
 * of this program consist of, and how few ALU packets could they fit in?
 * NVIDIA's bump-height shader has 51 slot operations in 13 packets (51 of
 * 52 slots used); each packet holds at most four, so ceil(operations / 4)
 * is a floor for any schedule of these operations. Operations whose result
 * only goes to the discard register R31, and plain copies, are counted
 * apart because they are the first candidates to remove or fold away.
 */
static void
debug_packing_bound(const struct grate_fp_shader *fp)
{
   unsigned packets = 0, ops = 0, discard = 0, copies = 0, mads = 0;
   unsigned hist[5] = {0};

   list_for_each_entry (struct fp_alu_instr_packet, packet,
                        &fp->alu_instructions, link) {
      unsigned used = 0;

      packets++;
      for (unsigned slot = 0; slot < 4; slot++) {
         const struct fp_alu_instr *alu = &packet->slots[slot];

         if (!alu->dst.write_low_sub_reg && !alu->dst.write_high_sub_reg)
            continue;
         used++;
         ops++;
         if (alu->dst.index == 31)
            discard++;
         else if (alu->src[1].index == 31 && alu->src[2].index == 31 &&
                  alu->src[3].index == 31)
            copies++;
         else
            mads++;
      }
      hist[used]++;
   }
   fprintf(stderr,
           "GRATE PACK: %u packets, %u slot ops (%u to R31 only, %u plain "
           "copies, %u other); packets with 1/2/3/4 ops: %u/%u/%u/%u; "
           "floor %u packets\n",
           packets, ops, discard, copies, mads, hist[1], hist[2], hist[3],
           hist[4], DIV_ROUND_UP(ops, 4));
}

/*
 * Dry run of a scheduler over slot operations (GRATE_PACKING_DEBUG, no effect
 * on the program): how many packets would a list scheduler need for the
 * ALU-only runs of steps, under different latency models? The steps that
 * carry a unit (interpolation or SFU, fetch, output write, sequencer
 * command, spill) keep their packets as they are.
 */
#define DRY_MAX_OPS 512

struct dry_op {
   int dst;
   unsigned wmask;          /* bit 0 low half, bit 1 high half */
   int src[4];
   unsigned rmask[4];
   unsigned height;
   int cycle;
};

static bool
dry_step_is_barrier(const struct fp_instr *instr)
{
   return instr->mfu_sched.num_instructions || (instr->tex & (1 << 10)) ||
          instr->dw.enable || instr->pseq || instr->spill_load;
}

/* Dependency latencies in packets: RAW raw_lat, WAR 0, WAW 1. */
static unsigned
dry_schedule(struct dry_op *ops, unsigned n, unsigned raw_lat, bool false_deps,
             bool raw_deps)
{
   /* edges stored as a dense matrix of latencies, -1 = none */
   static int8_t edge[DRY_MAX_OPS][DRY_MAX_OPS];
   for (unsigned i = 0; i < n; i++)
      for (unsigned j = 0; j < n; j++)
         edge[i][j] = -1;

   for (unsigned j = 0; j < n; j++) {
      for (unsigned i = 0; i < j; i++) {
         int best = -1;
         /* RAW: j reads what i writes */
         for (unsigned k = 0; k < 4; k++) {
            if (ops[j].src[k] >= 0 && ops[j].src[k] == ops[i].dst &&
                (ops[j].rmask[k] & ops[i].wmask) && raw_deps)
               best = MAX2(best, (int)raw_lat);
         }
         if (false_deps) {
            /* WAR: j writes what i reads (may share a packet) */
            for (unsigned k = 0; k < 4; k++)
               if (ops[i].src[k] >= 0 && ops[i].src[k] == ops[j].dst &&
                   (ops[i].rmask[k] & ops[j].wmask))
                  best = MAX2(best, 0);
            /* WAW: keep the order, the second write lands later */
            if (ops[i].dst >= 0 && ops[i].dst == ops[j].dst &&
                (ops[i].wmask & ops[j].wmask))
               best = MAX2(best, 1);
         }
         edge[i][j] = best;
      }
   }

   for (int i = (int)n - 1; i >= 0; i--) {
      ops[i].height = 1;
      for (unsigned j = i + 1; j < n; j++)
         if (edge[i][j] >= 0)
            ops[i].height = MAX2(ops[i].height, ops[j].height + edge[i][j]);
      ops[i].cycle = -1;
   }

   unsigned done = 0, cycle = 0;
   while (done < n && cycle < 4 * n + 8) {
      unsigned used = 0;
      bool progress = true;
      while (progress && used < 4) {
         progress = false;
         int pick = -1;
         for (unsigned j = 0; j < n; j++) {
            if (ops[j].cycle >= 0)
               continue;
            bool ready = true;
            for (unsigned i = 0; i < j && ready; i++)
               if (edge[i][j] >= 0 &&
                   (ops[i].cycle < 0 || ops[i].cycle + edge[i][j] > (int)cycle))
                  ready = false;
            if (!ready)
               continue;
            if (pick < 0 || ops[j].height > ops[pick].height)
               pick = j;
         }
         if (pick >= 0) {
            ops[pick].cycle = cycle;
            used++;
            done++;
            progress = true;
         }
      }
      cycle++;
   }
   return cycle;
}

static void
dry_fill_op(struct dry_op *op, const struct fp_alu_instr *alu)
{
   op->dst = alu->dst.index == 31 ? -1 : (int)alu->dst.index;
   op->wmask = (alu->dst.write_low_sub_reg ? 1 : 0) |
               (alu->dst.write_high_sub_reg ? 2 : 0);
   for (unsigned k = 0; k < 4; k++) {
      unsigned reg = alu->src[k].index;
      op->src[k] = (reg == 31 || (reg >= 24 && reg < 64)) ? -1 : (int)reg;
      op->rmask[k] = alu->src[k].datatype == FP_DATATYPE_FIXED10
                        ? (alu->src[k].sub_reg_select_high ? 2 : 1)
                        : 3;
   }
   /* a conditional write keeps the old value, so it reads it */
   if (alu->condition != FP_CONDITION_ALWAYS && op->dst >= 0) {
      for (unsigned k = 0; k < 4; k++)
         if (op->src[k] < 0) {
            op->src[k] = op->dst;
            op->rmask[k] = op->wmask;
            break;
         }
   }
}

static void
debug_schedule_bound(const struct grate_fp_shader *fp)
{
   unsigned num_packets = list_length(&fp->alu_instructions);
   const struct fp_alu_instr_packet **pk =
      CALLOC(num_packets + 1, sizeof(*pk));
   if (!pk)
      return;
   unsigned i = 0;
   list_for_each_entry (struct fp_alu_instr_packet, packet,
                        &fp->alu_instructions, link)
      pk[i++] = packet;

   static struct dry_op ops[DRY_MAX_OPS], all[DRY_MAX_OPS];
   unsigned num_all = 0;
   /* totals: fixed packets, then per model */
   unsigned fixed = 0, region_now = 0, ops_total = 0, skipped = 0;
   unsigned sum[4] = {0};
   unsigned n = 0, now = 0;
   bool in_region = false;

   list_for_each_entry (struct fp_instr, instr, &fp->fp_instructions, link) {
      bool barrier = dry_step_is_barrier(instr);
      for (unsigned p = 0; p < instr->alu_sched.num_instructions; p++) {
         unsigned address = instr->alu_sched.address + p;
         if (address >= num_packets)
            continue;
         for (unsigned slot = 0; slot < 4; slot++) {
            const struct fp_alu_instr *alu = &pk[address]->slots[slot];
            if ((alu->dst.write_low_sub_reg || alu->dst.write_high_sub_reg) &&
                num_all < DRY_MAX_OPS)
               dry_fill_op(&all[num_all++], alu);
         }
      }
      if (!barrier) {
         for (unsigned p = 0; p < instr->alu_sched.num_instructions; p++) {
            unsigned address = instr->alu_sched.address + p;
            if (address >= num_packets)
               continue;
            now++;
            for (unsigned slot = 0; slot < 4; slot++) {
               const struct fp_alu_instr *alu = &pk[address]->slots[slot];
               if (!alu->dst.write_low_sub_reg && !alu->dst.write_high_sub_reg)
                  continue;
               if (n >= DRY_MAX_OPS) {
                  skipped++;
                  continue;
               }
               dry_fill_op(&ops[n++], alu);
            }
         }
         in_region = true;
         continue;
      }
      /* a barrier step closes the region and keeps its own packets */
      if (in_region && n) {
         sum[0] += dry_schedule(ops, n, 2, true, true);
         sum[1] += dry_schedule(ops, n, 2, false, true);
         sum[2] += dry_schedule(ops, n, 1, true, true);
         sum[3] += dry_schedule(ops, n, 1, false, true);
         ops_total += n;
      }
      region_now += now;
      n = 0;
      now = 0;
      in_region = false;
      fixed += instr->alu_sched.num_instructions;
   }
   if (in_region && n) {
      sum[0] += dry_schedule(ops, n, 2, true, true);
      sum[1] += dry_schedule(ops, n, 2, false, true);
      sum[2] += dry_schedule(ops, n, 1, true, true);
      sum[3] += dry_schedule(ops, n, 1, false, true);
      ops_total += n;
   }
   region_now += now;

   fprintf(stderr,
           "GRATE SCHED DRY: %u packets now (%u in barrier steps, %u in "
           "ALU-only runs holding %u ops%s); same runs scheduled: lat 2 with "
           "all deps %u, lat 2 RAW only %u, forward lat 1 all deps %u, "
           "forward lat 1 RAW only %u; so the program would take %u/%u/%u/%u\n",
           fixed + region_now, fixed, region_now, ops_total,
           skipped ? ", some runs too long and cut" : "", sum[0], sum[1],
           sum[2], sum[3], fixed + sum[0], fixed + sum[1], fixed + sum[2],
           fixed + sum[3]);
   if (num_all)
      fprintf(stderr,
              "GRATE SCHED DRY: whole program as one run (optimistic: units "
              "and their results ignored) %u ops: lat 2 %u packets, forward "
              "lat 1 %u packets, floor %u\n",
              num_all, dry_schedule(all, num_all, 2, true, true),
              dry_schedule(all, num_all, 1, true, true),
              DIV_ROUND_UP(num_all, 4));
   FREE(pk);
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

/*
 * Writes the register writes that load a program into the sequencer, the
 * ALU, the MFU and the data writer: its steps and the instructions they
 * use, as the lists of one epoch.
 */
static void
emit_fp_blocks(struct util_dynarray *buf, enum drm_tegra_soc_id soc_id,
               struct list_head *steps, struct list_head *mfus,
               struct list_head *alus)
{
#define EPUSH(x) util_dynarray_append_typed(buf, uint32_t, (x))
   unsigned num_steps = list_length(steps);
   unsigned num_mfus = list_length(mfus);
   unsigned num_alus = list_length(alus);

   if (soc_id == DRM_TEGRA_SOC_T114) {
      // TODO: document/convert these values
      /* XXX: maybe not needed */
      EPUSH(host1x_opcode_incr(0x547, 0x0002));
      EPUSH(0xc0000000);
      EPUSH(0x00000000);
   }

   EPUSH(host1x_opcode_imm(REG_TGR3D_PSEQ_FLUSH, 0));

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_PSEQ_INST_DATA, num_steps));
   list_for_each_entry(struct fp_instr, instr, steps, link)
      EPUSH(instr->pseq);

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_AT_REMAP_DATA, num_steps));
   list_for_each_entry(struct fp_instr, instr, steps, link)
      EPUSH(grate_fp_pack_sched(&instr->mfu_sched));

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_AT_INST_DATA_LO, num_mfus * 2));
   list_for_each_entry(struct fp_mfu_instr, instr, mfus, link) {
      uint32_t words[2];
      grate_fp_pack_mfu(words, instr);
      EPUSH(words[0]);
      EPUSH(words[1]);
   }

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_TEX_INST_DATA, num_steps));
   list_for_each_entry(struct fp_instr, instr, steps, link)
      EPUSH(instr->tex);

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_REMAP_DATA, num_steps));
   list_for_each_entry(struct fp_instr, instr, steps, link) {
      if (soc_id == DRM_TEGRA_SOC_T114)
         EPUSH(grate_fp_pack_alu_sched_t114(&instr->alu_sched));
      else
         EPUSH(grate_fp_pack_sched(&instr->alu_sched));
   }

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_INST_DATA, num_alus * 4 * 2));
   list_for_each_entry(struct fp_alu_instr_packet, instr, alus, link) {
      for (int i = 0; i < 4; ++i) {
         uint32_t words[2];
         grate_fp_pack_alu(words, instr->slots + i);
         EPUSH(words[0]);
         EPUSH(words[1]);
      }
   }

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_ALU_P2CX_DATA, num_steps));
   list_for_each_entry(struct fp_instr, instr, steps, link)
      EPUSH(0x00000000);

   EPUSH(host1x_opcode_nonincr(REG_TGR3D_DW_INST_DATA, num_steps));
   list_for_each_entry(struct fp_instr, instr, steps, link)
      EPUSH(grate_fp_pack_dw(&instr->dw));
#undef EPUSH
}

/*
 * A fragment program that reads more than GRATE_FP_MAX_CONSTANTS distinct
 * constants does not fit the bank. Cut it into epochs at the borders of
 * its rounds (rows steps each), like NVIDIA's compiler: every epoch is a
 * program of its own, loaded at instruction 0, with the constants it reads
 * in bank slots from 0, and the registers of the pixels are kept between
 * the epochs. The result holds the blocks and the constants of every epoch,
 * see grate_fp_epochs. The steps and instructions of fp are consumed.
 */
#define GRATE_FP_LOGICAL_WORDS (GRATE_FP_MAX_LOGICAL_CONSTANTS / 32)
#define GRATE_FP_MAX_EPOCH_STEPS 64
/* the addresses of the ALU and MFU instructions of a step are six bits.
 * NVIDIA's compiler fills an epoch with exactly 64 ALU packets (addresses
 * 0-63, 512 words of ALU_INST_DATA) before it starts the next one; the MFU
 * limit is not measured (57 seen), 61 is kept. */
#define GRATE_FP_MAX_EPOCH_PACKETS 64
#define GRATE_FP_MAX_EPOCH_MFUS 61

static void
epoch_note_constants(const struct fp_alu_instr_packet *packet, uint32_t *bits)
{
   for (unsigned slot = 0; slot < 4; slot++) {
      for (unsigned i = 0; i < 4; i++) {
         unsigned index = packet->slots[slot].src[i].index;
         if (index >= 32 && index - 32 < GRATE_FP_MAX_LOGICAL_CONSTANTS)
            bits[(index - 32) / 32] |= 1u << ((index - 32) % 32);
      }
   }
}

static unsigned
epoch_count_bits(const uint32_t *bits)
{
   unsigned count = 0;
   for (unsigned i = 0; i < GRATE_FP_LOGICAL_WORDS; i++)
      count += util_bitcount(bits[i]);
   return count;
}

static void
free_epoch_lists(struct list_head *steps, struct list_head *mfus,
                 struct list_head *alus)
{
   list_for_each_entry_safe(struct fp_instr, step, steps, link)
      FREE(step);
   list_for_each_entry_safe(struct fp_mfu_instr, mfu, mfus, link)
      FREE(mfu);
   list_for_each_entry_safe(struct fp_alu_instr_packet, packet, alus, link)
      FREE(packet);
}

static bool
split_into_epochs(struct grate_fp_shader *fp, enum drm_tegra_soc_id soc_id,
                  bool has_spills, struct grate_fp_epochs **result)
{
   const unsigned rows = MAX2(fp->info.num_rows, 1);
   const unsigned num_steps = list_length(&fp->fp_instructions);
   const unsigned num_packets = list_length(&fp->alu_instructions);
   const unsigned num_mfus = list_length(&fp->mfu_instructions);
   const unsigned num_rounds = DIV_ROUND_UP(num_steps, rows);
   struct fp_instr **steps = CALLOC(num_steps + 1, sizeof(*steps));
   struct fp_alu_instr_packet **packets =
      CALLOC(num_packets + 1, sizeof(*packets));
   struct fp_mfu_instr **mfus = CALLOC(num_mfus + 1, sizeof(*mfus));
   uint32_t(*round_bits)[GRATE_FP_LOGICAL_WORDS] =
      CALLOC(num_rounds + 1, sizeof(*round_bits));
   unsigned *round_packets = CALLOC(num_rounds + 1, sizeof(unsigned));
   unsigned *round_mfus = CALLOC(num_rounds + 1, sizeof(unsigned));
   struct grate_fp_epochs *epochs = CALLOC_STRUCT(grate_fp_epochs);
   unsigned begin[GRATE_FP_MAX_EPOCHS + 1]; /* first round of each epoch */
   struct util_dynarray program;
   const char *why = NULL;
   unsigned count = 0;

   util_dynarray_init(&program, NULL);
   if (!steps || !packets || !mfus || !round_bits || !round_packets ||
       !round_mfus || !epochs) {
      why = "out of memory";
      goto fail;
   }
   if (fp->uses_derivatives || has_spills) {
      why = fp->uses_derivatives ? "derivatives need constant slots 0, 8, 16, 24"
                                 : "register spills";
      goto fail;
   }

   unsigned n = 0;
   list_for_each_entry(struct fp_instr, instr, &fp->fp_instructions, link)
      steps[n++] = instr;
   n = 0;
   list_for_each_entry(struct fp_alu_instr_packet, packet,
                       &fp->alu_instructions, link)
      packets[n++] = packet;
   n = 0;
   list_for_each_entry(struct fp_mfu_instr, mfu, &fp->mfu_instructions, link)
      mfus[n++] = mfu;

   /* Every step reads its own packets and MFU instructions, in order. */
   unsigned next_packet = 0, next_mfu = 0;
   for (unsigned i = 0; i < num_steps; i++) {
      const struct fp_sched *alu = &steps[i]->alu_sched;
      const struct fp_sched *mfu = &steps[i]->mfu_sched;

      if (alu->num_instructions) {
         if (alu->address < next_packet ||
             alu->address + alu->num_instructions > num_packets) {
            why = "steps share ALU instructions";
            goto fail;
         }
         next_packet = alu->address + alu->num_instructions;
         round_packets[i / rows] += alu->num_instructions;
         for (int j = 0; j < alu->num_instructions; j++)
            epoch_note_constants(packets[alu->address + j],
                                 round_bits[i / rows]);
      }
      if (mfu->num_instructions) {
         if (mfu->address < next_mfu ||
             mfu->address + mfu->num_instructions > num_mfus) {
            why = "steps share MFU instructions";
            goto fail;
         }
         next_mfu = mfu->address + mfu->num_instructions;
         round_mfus[i / rows] += mfu->num_instructions;
      }
   }

   /* Fill an epoch with rounds while their constants fit the bank. */
   uint32_t bits[GRATE_FP_LOGICAL_WORDS] = {0};
   unsigned epoch_steps = 0, epoch_packets = 0, epoch_mfus = 0;
   begin[0] = 0;
   for (unsigned r = 0; r < num_rounds; r++) {
      uint32_t merged[GRATE_FP_LOGICAL_WORDS];
      unsigned round_steps = MIN2(rows, num_steps - r * rows);

      if (epoch_count_bits(round_bits[r]) > GRATE_FP_MAX_CONSTANTS ||
          round_steps > GRATE_FP_MAX_EPOCH_STEPS ||
          round_packets[r] > GRATE_FP_MAX_EPOCH_PACKETS ||
          round_mfus[r] > GRATE_FP_MAX_EPOCH_MFUS) {
         why = "one round does not fit an epoch";
         goto fail;
      }
      for (unsigned i = 0; i < GRATE_FP_LOGICAL_WORDS; i++)
         merged[i] = bits[i] | round_bits[r][i];
      if (epoch_steps &&
          (epoch_count_bits(merged) > GRATE_FP_MAX_CONSTANTS ||
           epoch_steps + round_steps > GRATE_FP_MAX_EPOCH_STEPS ||
           epoch_packets + round_packets[r] > GRATE_FP_MAX_EPOCH_PACKETS ||
           epoch_mfus + round_mfus[r] > GRATE_FP_MAX_EPOCH_MFUS)) {
         if (++count == GRATE_FP_MAX_EPOCHS) {
            why = "more than five epochs";
            goto fail;
         }
         begin[count] = r;
         epoch_steps = epoch_packets = epoch_mfus = 0;
         memcpy(merged, round_bits[r], sizeof(merged));
      }
      memcpy(bits, merged, sizeof(bits));
      epoch_steps += round_steps;
      epoch_packets += round_packets[r];
      epoch_mfus += round_mfus[r];
   }
   begin[++count] = num_rounds;
   epochs->count = count;
   if (getenv("GRATE_EPOCH_DEBUG")) {
      fprintf(stderr, "GRATE epochs: %u steps, %u packets, %u mfu, %u rows, "
              "%u constants -> %u epochs\n", num_steps, num_packets, num_mfus,
              rows, fp->info.num_constants, count);
      for (unsigned e = 0; e < count; e++)
         fprintf(stderr, "  epoch %u: rounds %u..%u\n", e, begin[e],
                 begin[e + 1]);
   }

   for (unsigned e = 0; e < count; e++) {
      struct grate_fp_epoch *epoch = &epochs->epoch[e];
      const unsigned first = begin[e] * rows;
      const unsigned last = MIN2(begin[e + 1] * rows, num_steps);
      unsigned packet_first = ~0u, packet_end = 0;
      unsigned mfu_first = ~0u, mfu_end = 0;
      struct list_head epoch_steps_list, epoch_mfus, epoch_alus;
      int slot_of[GRATE_FP_MAX_LOGICAL_CONSTANTS];

      for (unsigned i = first; i < last; i++) {
         const struct fp_sched *alu = &steps[i]->alu_sched;
         const struct fp_sched *mfu = &steps[i]->mfu_sched;

         if (alu->num_instructions) {
            packet_first = MIN2(packet_first, alu->address);
            packet_end = MAX2(packet_end,
                              alu->address + alu->num_instructions);
         }
         if (mfu->num_instructions) {
            mfu_first = MIN2(mfu_first, mfu->address);
            mfu_end = MAX2(mfu_end, mfu->address + mfu->num_instructions);
         }
      }
      if (packet_first == ~0u)
         packet_first = packet_end = 0;
      if (mfu_first == ~0u)
         mfu_first = mfu_end = 0;

      list_inithead(&epoch_steps_list);
      list_inithead(&epoch_mfus);
      list_inithead(&epoch_alus);
      for (unsigned i = first; i < last; i++) {
         struct fp_instr *step = steps[i];

         step->alu_sched.address = step->alu_sched.num_instructions
                                      ? step->alu_sched.address - packet_first
                                      : 0;
         step->mfu_sched.address = step->mfu_sched.num_instructions
                                      ? step->mfu_sched.address - mfu_first
                                      : 0;
         list_del(&step->link);
         list_addtail(&step->link, &epoch_steps_list);
      }
      for (unsigned i = packet_first; i < packet_end; i++) {
         list_del(&packets[i]->link);
         list_addtail(&packets[i]->link, &epoch_alus);
      }
      for (unsigned i = mfu_first; i < mfu_end; i++) {
         list_del(&mfus[i]->link);
         list_addtail(&mfus[i]->link, &epoch_mfus);
      }

      /* The constants of this epoch take the bank slots from 0. */
      memset(slot_of, -1, sizeof(slot_of));
      list_for_each_entry(struct fp_alu_instr_packet, packet, &epoch_alus,
                          link) {
         for (unsigned slot = 0; slot < 4; slot++) {
            for (unsigned i = 0; i < 4; i++) {
               struct fp_alu_src_operand *src = &packet->slots[slot].src[i];
               unsigned id = src->index - 32;

               if (src->index < 32 || id >= GRATE_FP_MAX_LOGICAL_CONSTANTS)
                  continue;
               if (slot_of[id] < 0) {
                  if (epoch->num_constants == GRATE_FP_MAX_CONSTANTS) {
                     why = "an epoch reads more constants than the bank holds";
                     free_epoch_lists(&epoch_steps_list, &epoch_mfus,
                                      &epoch_alus);
                     goto fail;
                  }
                  slot_of[id] = epoch->num_constants;
                  epoch->constants[epoch->num_constants].word =
                     fp->info.constants[id].word;
                  epoch->constants[epoch->num_constants].immediate =
                     fp->info.constants[id].immediate;
                  epoch->num_constants++;
               }
               src->index = 32 + slot_of[id];
            }
         }
      }

      /* The block: the offsets of the instruction memories are reset like in
       * NVIDIA's blocks, then the program. Blocks start on 16 bytes. */
      struct util_dynarray body;
      util_dynarray_init(&body, NULL);
      util_dynarray_append_typed(&body, uint32_t,
                                 host1x_opcode_incr(
                                    REG_TGR3D_GLOBAL_INST_OFFSET, 1));
      util_dynarray_append_typed(&body, uint32_t, 0);
      util_dynarray_append_typed(&body, uint32_t,
                                 host1x_opcode_incr(REG_TGR3D_AT_INST_OFFSET,
                                                    1));
      util_dynarray_append_typed(&body, uint32_t, 0);
      util_dynarray_append_typed(&body, uint32_t,
                                 host1x_opcode_incr(REG_TGR3D_ALU_INST_OFFSET,
                                                    1));
      util_dynarray_append_typed(&body, uint32_t, 0);
      emit_fp_blocks(&body, soc_id, &epoch_steps_list, &epoch_mfus,
                     &epoch_alus);

      while (program.size % 16)
         util_dynarray_append_typed(&program, uint32_t, 0);
      epoch->prog_offset = program.size;
      epoch->prog_words = body.size / sizeof(uint32_t);
      epoch->steps = last - first;
      epoch->packets = packet_end - packet_first;
      epoch->const_bank = 0;
      util_dynarray_append_array(&program, uint32_t,
                                 (uint32_t *)body.data, epoch->prog_words);
      util_dynarray_fini(&body);
      free_epoch_lists(&epoch_steps_list, &epoch_mfus, &epoch_alus);
   }

   epochs->program = MALLOC(program.size);
   if (!epochs->program) {
      why = "out of memory";
      goto fail;
   }
   memcpy(epochs->program, program.data, program.size);
   epochs->program_bytes = program.size;
   fp->info.num_constants = 0;
   *result = epochs;
   epochs = NULL;
   util_dynarray_fini(&program);
   FREE(steps);
   FREE(packets);
   FREE(mfus);
   FREE(round_bits);
   FREE(round_packets);
   FREE(round_mfus);
   return true;

fail:
   if (why)
      fprintf(stderr, "GRATE FRAG: scalar constant bank exhausted (%s; %u steps, %u packets, %u mfu, %u rows, %u constants)\n", why, num_steps, num_packets, num_mfus, rows, fp->info.num_constants);
   util_dynarray_fini(&program);
   FREE(steps);
   FREE(packets);
   FREE(mfus);
   FREE(round_bits);
   FREE(round_packets);
   FREE(round_mfus);
   FREE(epochs);
   return false;
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

   if (getenv("GRATE_PACKING_DEBUG")) {
      debug_packing_bound(&fp);
      debug_schedule_bound(&fp);
   }
   if (fp.spilling)
      coalesce_scalar_alu(&fp);
   align_to_rows(&fp);

   unsigned num_fp_instrs = list_length(&fp.fp_instructions);
   unsigned num_mfu_instrs = list_length(&fp.mfu_instructions);
   unsigned num_alu_instrs = list_length(&fp.alu_instructions);
   if (getenv("GRATE_PACKING_DEBUG"))
      fprintf(stderr, "GRATE SCHED DRY: emitted after align_to_rows: %u ALU packets in %u steps (%u rows)\n", num_alu_instrs, num_fp_instrs, fp.info.num_rows);
   /* The ALU and MFU instruction addresses of a step are six bits: a
    * program with more instructions than that also needs epochs. */
   if (!fp.allocation_failed &&
       (fp.info.num_constants > GRATE_FP_MAX_CONSTANTS ||
        num_alu_instrs > GRATE_FP_MAX_EPOCH_PACKETS ||
        num_mfu_instrs > GRATE_FP_MAX_EPOCH_MFUS) &&
       !split_into_epochs(&fp, soc_id, has_spills, &fp.info.epochs)) {
      free_fragment_ir(&fp);
      return false;
   }
   if (fp.info.epochs) {
      /* every epoch was checked against the limits of one */
   } else if (fp.allocation_failed || num_fp_instrs > 64 ||
              num_mfu_instrs >= 64) {
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
   if (getenv("GRATE_NUMROWS")) /* EXPERIMENT hook, see docs/grate-alu-latency.md */
      fp.info.num_rows = atoi(getenv("GRATE_NUMROWS"));
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
   if (getenv("GRATE_REPLAY_N")) /* EXPERIMENT hook */
      num_fp_instrs = atoi(getenv("GRATE_REPLAY_N"));
   PUSH(0x20006000 | num_fp_instrs);

   PUSH(host1x_opcode_incr(REG_TGR3D_PSEQ_DWR_IF_STATE, 1));
   // TODO: document/convert these values
   PUSH(TGR3D_PSEQ_DWR_IF_STATE_COUNT(num_fp_instrs));

   size_t replay_start = buf.size;
   emit_fp_blocks(&buf, soc_id, &fp.fp_instructions, &fp.mfu_instructions,
                   &fp.alu_instructions);

   if (getenv("GRATE_REPLAY")) { /* EXPERIMENT hook: a raw stream replaces our program */
      FILE *rf = fopen(getenv("GRATE_REPLAY"), "rb");
      if (rf) {
         uint32_t rw[1 << 14];
         size_t n = fread(rw, 4, 1 << 14, rf);
         fclose(rf);
         buf.size = replay_start;
         for (size_t i = 0; i < n; i++)
            PUSH(rw[i]);
      }
   }

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
      if (variant->epochs) {
         grate_bo_unref(variant->epochs->program_bo);
         grate_bo_unref(variant->epochs->constant_bo);
         FREE(variant->epochs->program);
         FREE(variant->epochs);
      }
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
