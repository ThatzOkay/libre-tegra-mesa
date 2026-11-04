#include "grate_common.h"
#include "grate_compiler.h"
#include "fpir.h"

#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_util.h"
#include "util/u_dynarray.h"

#include "util/u_memory.h"

static struct fp_alu_src_operand
fp_alu_src_row(int index)
{
   assert(index >= 0 && index < 16);
   struct fp_alu_src_operand src = {
      .index = index
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_zero()
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 0
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_one()
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 1
   };
   return src;
}

static struct fp_alu_instr
fp_alu_sMOV(struct fp_alu_dst_operand dst, struct fp_alu_src_operand src)
{
   struct fp_alu_instr ret = {
      .op = FP_ALU_OP_MAD,
      .dst = dst,
      .src = {
         src,
         fp_alu_src_one(),
         fp_alu_src_zero(),
         fp_alu_src_one()
      }
   };
   return ret;
}

#define TEMP_FX10 0x100
#define TEMP_HIGH 0x200

/* Comparison results have their own boolean bank. Sampled UNORM colours
 * retain their exact FX10 representation through MOV/CMP, while arithmetic
 * uses FP20. R0-R3 become available after the last interpolation/TEX fetch.
 */
static int
allocate_scalar(struct grate_fp_shader *fp, unsigned kind)
{
   if (kind == 2) {
      for (unsigned reg = 64; reg < 72; reg++) {
         if (!fp->occupied[reg]) {
            fp->occupied[reg] = fp->claimed[reg] = 1;
            return reg | TEMP_FX10;
         }
      }
   } else {
      const unsigned globals[] = {16,17,18,19,20,21,22,23};
      const unsigned rows[] = {0,1,2,3};
      const unsigned *pool = fp->late_rows ? rows : globals;
      unsigned pool_size = fp->late_rows ? ARRAY_SIZE(rows) : ARRAY_SIZE(globals);
      /* Prefer an operand whose last read is this scalar instruction. */
      for (unsigned n = 0; n < pool_size; n++) {
         unsigned reg = pool[n];
         if (!kind && fp->reusable[reg] == 3 && !fp->claimed[reg]) {
            fp->occupied[reg] = fp->claimed[reg] = 3;
            return reg;
         }
         if (kind == 1) {
            for (unsigned half = 0; half < 2; half++) {
               unsigned mask = 1 << half;
               if ((fp->reusable[reg] & mask) && !(fp->claimed[reg] & mask)) {
                  fp->occupied[reg] |= mask;
                  fp->claimed[reg] |= mask;
                  return reg | TEMP_FX10 | (half ? TEMP_HIGH : 0);
               }
            }
         }
      }
      for (unsigned n = 0; n < pool_size; n++) {
         unsigned reg = pool[n];
         if (!kind && !fp->occupied[reg]) {
            fp->occupied[reg] = fp->claimed[reg] = 3;
            return reg;
         }
         if (kind == 1) {
            for (unsigned half = 0; half < 2; half++) {
               unsigned mask = 1 << half;
               if (!(fp->occupied[reg] & mask)) {
                  fp->occupied[reg] |= mask;
                  fp->claimed[reg] |= mask;
                  return reg | TEMP_FX10 | (half ? TEMP_HIGH : 0);
               }
            }
         }
      }
   }
   if (!fp->allocation_failed)
      fprintf(stderr, "GRATE FRAG: scalar register bank exhausted (spilling unsupported)\n");
   fp->allocation_failed = true;
   return 31;
}

static struct fp_alu_dst_operand
fp_alu_dst(struct grate_fp_shader *fp, const struct tgsi_dst_register *dst, int subreg, bool saturate)
{
   struct fp_alu_dst_operand ret = { 0 };


   if (dst->File == TGSI_FILE_OUTPUT) {
      ret.index = 2; // HACK: r2+r3 to match hard-coded store shader for now

      // fixed10
      // swizzle RGBA -> BGRA
      int o = subreg < 3 ? (2 - subreg) : 3;
      ret.index += o / 2;
      ret.write_low_sub_reg = (o % 2) == 0;
      ret.write_high_sub_reg = (o % 2) != 0;
   } else {
      assert(dst->File == TGSI_FILE_TEMPORARY && dst->Index <= GRATE_FP_MAX_TEMPS);
      if (fp->temp_dst[dst->Index][subreg] < 0) {
         /* Internal interpolation staging lives only within this instruction. */
         assert(dst->Index == fp->num_temporaries);
         fp->temp_dst[dst->Index][subreg] = allocate_scalar(fp, false);
         fp->temp_src[dst->Index][subreg] = fp->temp_dst[dst->Index][subreg];
      }
      int location = fp->temp_dst[dst->Index][subreg];
      ret.index = location & 0x7f;
      ret.write_low_sub_reg = !(location & TEMP_HIGH);
      ret.write_high_sub_reg = !(location & TEMP_FX10) || (location & TEMP_HIGH);
   }

   ret.saturate = saturate;

   return ret;
}

static struct fp_alu_src_operand
fp_constant(struct grate_fp_shader *fp, int word, float immediate)
{
   unsigned i;
   for (i = 0; i < fp->info.num_constants; i++) {
      if (fp->info.constants[i].word == word &&
          (word >= 0 || fp->info.constants[i].immediate == immediate))
         break;
   }
   assert(i < 32);
   if (i == fp->info.num_constants) {
      fp->info.constants[i].word = word;
      fp->info.constants[i].immediate = immediate;
      fp->info.num_constants++;
   }
   struct fp_alu_src_operand value = {.index = 32 + i};
   return value;
}

static struct fp_alu_src_operand
fp_tgsi_src(struct grate_fp_shader *fp, const struct tgsi_src_register *src,
            unsigned component, struct fp_mfu_instr *mfu)
{
   unsigned swizzle[] = {src->SwizzleX, src->SwizzleY, src->SwizzleZ, src->SwizzleW};
   unsigned comp = swizzle[component];
   struct fp_alu_src_operand value;

   switch (src->File) {
   case TGSI_FILE_INPUT:
      /* Interpolate the selected source channel, not the destination channel. */
      assert(mfu);
      assert(mfu->var[comp].op == FP_VAR_OP_NOP || mfu->var[comp].tram_row == src->Index);
      mfu->var[comp].op = FP_VAR_OP_FP20;
      mfu->var[comp].tram_row = src->Index;
      fp->info.max_tram_row = MAX2(fp->info.max_tram_row, src->Index + 1);
      value = fp_alu_src_row(comp);
      break;
   case TGSI_FILE_IMMEDIATE: {
      float imm = fp->immediates[src->Index][comp];
      value = imm == 1.0f ? fp_alu_src_one() :
              imm == 0.0f ? fp_alu_src_zero() : fp_constant(fp, -1, imm);
      break;
   }
   case TGSI_FILE_CONSTANT:
      assert(!src->Indirect);
      value = fp_constant(fp, src->Index * 4 + comp, 0.0f);
      break;
   case TGSI_FILE_TEMPORARY:
      assert(src->Index <= GRATE_FP_MAX_TEMPS);
      int reg = fp->temp_src[src->Index][comp];
      assert(reg >= 0 || fp->allocation_failed);
      value = fp_alu_src_row(0);
      value.index = reg >= 0 ? reg & 0x7f : 31;
      value.datatype = (reg & TEMP_FX10) ? FP_DATATYPE_FIXED10 : FP_DATATYPE_FP20;
      value.sub_reg_select_high = !!(reg & TEMP_HIGH);
      break;
   default:
      UNREACHABLE("unsupported fragment ALU source");
   }
   value.absolute_value = src->Absolute;
   value.negate = src->Negate;
   return value;
}

static void
setup_interpolation(struct fp_mfu_instr *mfu)
{
   mfu->sfu.op = FP_SFU_OP_RCP;
   mfu->sfu.reg = 4;
   for (unsigned k = 0; k < 2; k++) {
      mfu->mul[k].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
      mfu->mul[k].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
      mfu->mul[k].src[1] = k ? FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1 : FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;
   }
}

static void
emit_vALU(struct grate_fp_shader *fp, const struct tgsi_dst_register *dst,
          bool saturate, const struct tgsi_src_register *src,
          const struct tgsi_src_register *rhs,
          const struct tgsi_src_register *third, unsigned opcode)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);

   struct fp_mfu_instr *mfu = NULL;
   if (src->File == TGSI_FILE_INPUT || (rhs && rhs->File == TGSI_FILE_INPUT) ||
       (third && third->File == TGSI_FILE_INPUT)) {
      mfu = CALLOC_STRUCT(fp_mfu_instr);
      list_inithead(&mfu->link);
      setup_interpolation(mfu);
   }

   bool output_stage = false;
   const struct tgsi_src_register *operands[] = {src, rhs, third};
   if (dst->File == TGSI_FILE_OUTPUT) {
      for (unsigned s = 0; s < 3; s++) {
         const struct tgsi_src_register *operand = operands[s];
         if (!operand || operand->File != TGSI_FILE_TEMPORARY) continue;
         unsigned swizzle[] = {operand->SwizzleX, operand->SwizzleY, operand->SwizzleZ, operand->SwizzleW};
         for (unsigned c = 0; c < 4; c++) {
            int loc = fp->temp_src[operand->Index][swizzle[c]];
            if ((dst->WriteMask & (1 << c)) && loc >= 0 && (loc & 0x7f) < 4)
               output_stage = true;
         }
      }
   }
   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   int alu_instrs = 0;
   list_inithead(&alu->link);
   for (int i = 0; i < 4; ++i) {
      if ((dst->WriteMask & (1 << i)) == 0)
         continue;

      struct fp_alu_src_operand a = fp_tgsi_src(fp, src, i, mfu);
      struct fp_alu_src_operand b = rhs ? fp_tgsi_src(fp, rhs, i, mfu) : fp_alu_src_one();
      struct fp_alu_instr op = fp_alu_sMOV(fp_alu_dst(fp, dst, i, saturate), a);
      op.src[1] = b;
      switch (opcode) {
      case TGSI_OPCODE_ADD:
         op.src[1] = fp_alu_src_one();
         op.src[2] = b;
         break;
      case TGSI_OPCODE_SLT:
         /* SLT(a,b) = (b-a > 0). */
         op.src[0] = b;
         op.src[1] = fp_alu_src_one();
         op.src[2] = a;
         op.src[2].negate = !a.negate;
         op.condition = FP_CONDITION_GREATER;
         break;
      case TGSI_OPCODE_SGE:
      case TGSI_OPCODE_SEQ:
         op.src[1] = fp_alu_src_one();
         op.src[2] = b;
         op.src[2].negate = !b.negate;
         op.condition = opcode == TGSI_OPCODE_SGE ? FP_CONDITION_GEQUAL : FP_CONDITION_EQUAL;
         break;
      case TGSI_OPCODE_CMP:
         op.op = FP_ALU_OP_CSEL;
         op.src[2] = fp_tgsi_src(fp, third, i, mfu);
         break;
      default:
         break;
      }
      if (output_stage) {
         op.dst.index = 16 + i;
         op.dst.write_low_sub_reg = true;
         op.dst.write_high_sub_reg = true;
      }
      alu->slots[alu_instrs++] = op;
   }
   inst->alu_sched.num_instructions = 1;
   inst->alu_sched.address = list_length(&fp->alu_instructions);

   if (mfu != NULL) {
      inst->mfu_sched.num_instructions = 1;
      inst->mfu_sched.address = list_length(&fp->mfu_instructions);
      list_addtail(&mfu->link, &fp->mfu_instructions);
   }

   if (dst->File == TGSI_FILE_OUTPUT) {
      inst->dw.enable = 1;
      /* Hardware target 0 is reserved for depth, with or without a
       * depth attachment. Match grate_set_framebuffer_state().
       */
      inst->dw.index = dst->Index + 1;
      inst->dw.stencil_write = 0;
      inst->dw.src_regs = FP_DW_REGS_R2_R3; // hard-coded for now
   }

   list_addtail(&alu->link, &fp->alu_instructions);
   if (output_stage) {
      struct fp_alu_instr_packet *nop = CALLOC_STRUCT(fp_alu_instr_packet);
      struct fp_alu_instr_packet *store = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&nop->link);
      list_inithead(&store->link);
      unsigned slot = 0;
      for (unsigned c = 0; c < 4; c++) {
         struct fp_alu_dst_operand discard = {.index = 31};
         nop->slots[c] = fp_alu_sMOV(discard, fp_alu_src_zero());
         if (!(dst->WriteMask & (1 << c))) continue;
         struct fp_alu_src_operand value = {.index = 16 + c};
         store->slots[slot++] = fp_alu_sMOV(fp_alu_dst(fp, dst, c, saturate), value);
      }
      list_addtail(&nop->link, &fp->alu_instructions);
      list_addtail(&store->link, &fp->alu_instructions);
      inst->alu_sched.num_instructions = 3;
   }
   list_addtail(&inst->link, &fp->fp_instructions);
}

/* The initial TEX path supports an interpolated vec2 coordinate and one
 * base-level 2D sample. TEX returns packed fx10 channels in row registers 2/3.
 */
static void
emit_TEX(struct grate_fp_shader *fp, const struct tgsi_full_instruction *tgsi)
{
   const struct tgsi_src_register *coord = &tgsi->Src[0].Register;
   const struct tgsi_src_register *sampler = &tgsi->Src[1].Register;
   const struct tgsi_dst_register *dst = &tgsi->Dst[0].Register;
   assert(tgsi->Texture.Texture == TGSI_TEXTURE_2D);
   assert(coord->File == TGSI_FILE_INPUT && !coord->Negate && !coord->Absolute);
   assert(coord->SwizzleX == 0 && coord->SwizzleY == 1);
   assert(sampler->File == TGSI_FILE_SAMPLER && sampler->Index < 16);

   struct fp_instr *interp = CALLOC_STRUCT(fp_instr);
   struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
   list_inithead(&interp->link);
   list_inithead(&mfu->link);
   setup_interpolation(mfu);
   for (unsigned i = 0; i < 2; i++) {
      mfu->var[i].op = FP_VAR_OP_FP20;
      mfu->var[i].tram_row = coord->Index;
   }
   fp->info.max_tram_row = MAX2(fp->info.max_tram_row, coord->Index + 1);
   interp->mfu_sched.address = list_length(&fp->mfu_instructions);
   interp->mfu_sched.num_instructions = 1;
   list_addtail(&mfu->link, &fp->mfu_instructions);
   list_addtail(&interp->link, &fp->fp_instructions);

   struct fp_instr *sample = CALLOC_STRUCT(fp_instr);
   list_inithead(&sample->link);
   sample->tex = (1 << 10) | (1 << 5) | sampler->Index;
   if (dst->File == TGSI_FILE_TEMPORARY && fp->texture_rows) {
      list_addtail(&sample->link, &fp->fp_instructions);
      return;
   }
   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   list_inithead(&alu->link);
   unsigned slot = 0;
   for (unsigned i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i))) continue;
      unsigned channel = i;
      struct fp_alu_src_operand src = {
         .index = 2 + channel / 2,
         .datatype = FP_DATATYPE_FIXED10,
         .sub_reg_select_high = channel % 2,
      };
      alu->slots[slot++] = fp_alu_sMOV(fp_alu_dst(fp, dst, i, tgsi->Instruction.Saturate), src);
   }
   sample->alu_sched.address = list_length(&fp->alu_instructions);
   sample->alu_sched.num_instructions = 1;
   list_addtail(&alu->link, &fp->alu_instructions);
   if (dst->File == TGSI_FILE_OUTPUT) {
      sample->dw.enable = true;
      sample->dw.index = dst->Index + 1;
      sample->dw.src_regs = FP_DW_REGS_R2_R3;
   }
   list_addtail(&sample->link, &fp->fp_instructions);
}

static void
emit_tgsi_instr(struct grate_fp_shader *fp, const struct tgsi_full_instruction *inst)
{
   bool saturate = inst->Instruction.Saturate != 0;
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      if (inst->Src[i].Register.File == TGSI_FILE_CONSTANT && inst->Src[i].Register.Dimension)
         assert(inst->Src[i].Dimension.Index == 0 && !inst->Src[i].Dimension.Indirect);
   }

   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_TEX:
      emit_TEX(fp, inst);
      break;
   case TGSI_OPCODE_MOV:
      emit_vALU(fp, &inst->Dst[0].Register, saturate,
                    &inst->Src[0].Register, NULL, NULL, TGSI_OPCODE_MOV);
      break;

   case TGSI_OPCODE_ADD:
   case TGSI_OPCODE_SLT:
   case TGSI_OPCODE_SGE:
   case TGSI_OPCODE_SEQ:
   case TGSI_OPCODE_MUL: {
      const struct tgsi_src_register *a = &inst->Src[0].Register;
      const struct tgsi_src_register *b = &inst->Src[1].Register;
      struct tgsi_src_register staged = {0};
      if (a->File == TGSI_FILE_INPUT && b->File == TGSI_FILE_INPUT && a->Index != b->Index) {
         /* One MFU packet interpolates one vec4. Stage A in allocator-owned
          * scratch scalars before loading B. */
         assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
         struct tgsi_dst_register scratch = {0};
         scratch.File = TGSI_FILE_TEMPORARY;
         scratch.Index = fp->num_temporaries;
         scratch.WriteMask = TGSI_WRITEMASK_XYZW;
         emit_vALU(fp, &scratch, false, a, NULL, NULL, TGSI_OPCODE_MOV);
         staged.File = TGSI_FILE_TEMPORARY;
         staged.Index = fp->num_temporaries;
         staged.SwizzleX = TGSI_SWIZZLE_X;
         staged.SwizzleY = TGSI_SWIZZLE_Y;
         staged.SwizzleZ = TGSI_SWIZZLE_Z;
         staged.SwizzleW = TGSI_SWIZZLE_W;
         a = &staged;
      }
      emit_vALU(fp, &inst->Dst[0].Register, saturate, a, b, NULL, inst->Instruction.Opcode);

      break;
   }

   case TGSI_OPCODE_CMP: {
      const struct tgsi_src_register *a = &inst->Src[0].Register;
      const struct tgsi_src_register *b = &inst->Src[1].Register;
      const struct tgsi_src_register *c = &inst->Src[2].Register;
      struct tgsi_src_register staged = {0};
      /* An MFU packet can fetch one varying row. Preserve the first of
       * two different rows before fetching the second for selection. */
      const struct tgsi_src_register **inputs[] = {&a, &b, &c};
      for (unsigned i = 0; i < 3; i++) {
         bool needs_stage = false;
         for (unsigned j = i + 1; j < 3; j++)
            needs_stage |= (*inputs[i])->File == TGSI_FILE_INPUT &&
                           (*inputs[j])->File == TGSI_FILE_INPUT &&
                           (*inputs[i])->Index != (*inputs[j])->Index;
         if (!needs_stage) continue;
         assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS && !staged.File);
         struct tgsi_dst_register scratch = {0};
         scratch.File = TGSI_FILE_TEMPORARY;
         scratch.Index = fp->num_temporaries;
         scratch.WriteMask = TGSI_WRITEMASK_XYZW;
         emit_vALU(fp, &scratch, false, *inputs[i], NULL, NULL, TGSI_OPCODE_MOV);
         staged.File = TGSI_FILE_TEMPORARY;
         staged.Index = scratch.Index;
         staged.SwizzleX = TGSI_SWIZZLE_X;
         staged.SwizzleY = TGSI_SWIZZLE_Y;
         staged.SwizzleZ = TGSI_SWIZZLE_Z;
         staged.SwizzleW = TGSI_SWIZZLE_W;
         *inputs[i] = &staged;
      }
      emit_vALU(fp, &inst->Dst[0].Register, saturate, a, b, c, TGSI_OPCODE_CMP);
      break;
   }

   default:
      fprintf(stderr, "GRATE FRAG TGSI UNIMPLEMENTED: 0x%02x\n", inst->Instruction.Opcode);
      assert(0);
   }
}

#define LINK_SRC(index) ((index) << 3)
#define LINK_DST(index, comp, type) (((comp) | (type) << 2) << ((index) * 4))
#define LINK_DST_NONE      0
#define LINK_DST_FX10_LOW  1
#define LINK_DST_FX10_HIGH 2
#define LINK_DST_FP20      3

static void
emit_tgsi_input(struct grate_fp_shader *fp, const struct tgsi_full_declaration *decl)
{
   grate_trace();
   assert(decl->Range.First == decl->Range.Last);

   uint32_t src = LINK_SRC(decl->Range.First + 1) | (decl->Range.First << 9);
   uint32_t dst = 0;
   for (int i = 0; i < 4; ++i)
      dst |= LINK_DST(i, i, LINK_DST_FP20);

   fp->info.inputs[fp->info.num_inputs].src = src;
   fp->info.inputs[fp->info.num_inputs].dst = dst;

   if (decl->Declaration.Semantic == TGSI_SEMANTIC_COLOR)
      fp->info.color_input = decl->Range.First;

   fp->info.num_inputs++;
}

static void
emit_tgsi_declaration(struct grate_fp_shader *fp, const struct tgsi_full_declaration *decl)
{
   switch (decl->Declaration.File) {
   case TGSI_FILE_TEMPORARY:
      fp->num_temporaries = MAX2(fp->num_temporaries, decl->Range.Last + 1);
      break;
   case TGSI_FILE_INPUT:
      emit_tgsi_input(fp, decl);
      break;
   }
}

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi)
{
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);

   fp->info.num_inputs = 0;
   fp->info.num_constants = 0;
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;
   fp->num_immediates = 0;
   fp->num_temporaries = 0;
   fp->allocation_failed = false;
   memset(fp->occupied, 0, sizeof(fp->occupied));
   memset(fp->temp_src, -1, sizeof(fp->temp_src));
   memset(fp->temp_dst, -1, sizeof(fp->temp_dst));
   struct util_dynarray instructions;
   util_dynarray_init(&instructions, NULL);

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_IMMEDIATE: {
         const struct tgsi_full_immediate *imm = &tgsi->FullToken.FullImmediate;
         unsigned n = imm->Immediate.NrTokens - 1; /* number of 32-bit values: 1 to 4 */

         assert(fp->num_immediates < GRATE_FP_MAX_IMMEDIATES);
         for (unsigned i = 0; i < 4; ++i)
            fp->immediates[fp->num_immediates][i] = i < n ? imm->u[i].Float : 0.0f;
         fp->num_immediates++;
         break;
      }

      case TGSI_TOKEN_TYPE_DECLARATION:
         emit_tgsi_declaration(fp, &tgsi->FullToken.FullDeclaration);
         break;

      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode != TGSI_OPCODE_END)
            util_dynarray_append_typed(&instructions, struct tgsi_full_instruction, tgsi->FullToken.FullInstruction);
         break;
      }
   }

   /* Fragment output values stay virtual until all shader operations have
    * consumed the sampled pixel. Early masked DW stores alias R2/R3 and
    * would destroy still-live texture channels. */
   int output_temps[16];
   unsigned output_masks[16] = {0};
   memset(output_temps, -1, sizeof(output_temps));
   util_dynarray_foreach(&instructions, struct tgsi_full_instruction, instruction) {
      for (unsigned d = 0; d < instruction->Instruction.NumDstRegs; d++) {
         struct tgsi_dst_register *dst = &instruction->Dst[d].Register;
         if (dst->File != TGSI_FILE_OUTPUT) continue;
         unsigned index = dst->Index;
         assert(index < 16);
         if (output_temps[index] < 0)
            output_temps[index] = fp->num_temporaries++;
         output_masks[index] |= dst->WriteMask;
         dst->File = TGSI_FILE_TEMPORARY;
         dst->Index = output_temps[index];
      }
   }
   for (unsigned out = 0; out < 16; out++) {
      if (output_temps[out] < 0) continue;
      struct tgsi_full_instruction store = {0};
      store.Instruction.Opcode = TGSI_OPCODE_MOV;
      store.Instruction.NumDstRegs = 1;
      store.Instruction.NumSrcRegs = 1;
      store.Dst[0].Register.File = TGSI_FILE_OUTPUT;
      store.Dst[0].Register.Index = out;
      store.Dst[0].Register.WriteMask = output_masks[out];
      store.Src[0].Register.File = TGSI_FILE_TEMPORARY;
      store.Src[0].Register.Index = output_temps[out];
      store.Src[0].Register.SwizzleX = TGSI_SWIZZLE_X;
      store.Src[0].Register.SwizzleY = TGSI_SWIZZLE_Y;
      store.Src[0].Register.SwizzleZ = TGSI_SWIZZLE_Z;
      store.Src[0].Register.SwizzleW = TGSI_SWIZZLE_W;
      util_dynarray_append_typed(&instructions, struct tgsi_full_instruction, store);
   }
   assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
   unsigned count = util_dynarray_num_elements(&instructions, struct tgsi_full_instruction);
   struct tgsi_full_instruction *code = instructions.data;
   uint8_t (*live_after)[GRATE_FP_MAX_TEMPS] = calloc(count + 1, sizeof(*live_after));
   assert(live_after);
   int last_fetch = -1;
   for (int n = (int)count - 1; n >= 0; n--) {
      memcpy(live_after[n], live_after[n + 1], sizeof(*live_after));
      const struct tgsi_full_instruction *inst = &code[n];
      if (inst->Instruction.Opcode == TGSI_OPCODE_TEX && last_fetch < 0)
         last_fetch = n;
      for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
         const struct tgsi_dst_register *dst = &inst->Dst[d].Register;
         if (dst->File == TGSI_FILE_TEMPORARY)
            live_after[n][dst->Index] &= ~dst->WriteMask;
      }
      for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++) {
         const struct tgsi_src_register *reg = &inst->Src[src].Register;
         if (reg->File == TGSI_FILE_INPUT && last_fetch < 0)
            last_fetch = n;
         if (reg->File == TGSI_FILE_TEMPORARY)
            live_after[n][reg->Index] |= tgsi_util_get_inst_usage_mask(inst, src);
      }
   }
   for (unsigned n = 0; n < count; n++) {
      const struct tgsi_full_instruction *inst = &code[n];
      fp->late_rows = (int)n > last_fetch;
      fp->texture_rows = (int)n == last_fetch && inst->Instruction.Opcode == TGSI_OPCODE_TEX;
      memset(fp->claimed, 0, sizeof(fp->claimed));
      memset(fp->reusable, 0, sizeof(fp->reusable));
      memcpy(fp->temp_dst, fp->temp_src, sizeof(fp->temp_dst));
      memset(fp->temp_dst[fp->num_temporaries], -1, sizeof(fp->temp_dst[0]));
      bool predicate = inst->Instruction.Opcode == TGSI_OPCODE_SLT ||
                       inst->Instruction.Opcode == TGSI_OPCODE_SGE ||
                       inst->Instruction.Opcode == TGSI_OPCODE_SEQ;
      for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
         const struct tgsi_dst_register *dst = &inst->Dst[d].Register;
         if (dst->File != TGSI_FILE_TEMPORARY) continue;
         for (unsigned c = 0; c < 4; c++) {
            if (!(dst->WriteMask & (1 << c))) continue;
            unsigned kind = predicate ? 2 : inst->Instruction.Opcode == TGSI_OPCODE_TEX ? 1 : 0;
            if (inst->Instruction.Opcode == TGSI_OPCODE_MOV || inst->Instruction.Opcode == TGSI_OPCODE_CMP) {
               bool packed = true;
               unsigned first = inst->Instruction.Opcode == TGSI_OPCODE_CMP ? 1 : 0;
               for (unsigned src = first; src < inst->Instruction.NumSrcRegs; src++) {
                  const struct tgsi_src_register *operand = &inst->Src[src].Register;
                  unsigned swizzle[] = {operand->SwizzleX, operand->SwizzleY, operand->SwizzleZ, operand->SwizzleW};
                  packed &= operand->File == TGSI_FILE_TEMPORARY &&
                            fp->temp_src[operand->Index][swizzle[c]] >= 0 &&
                            (fp->temp_src[operand->Index][swizzle[c]] & TEMP_FX10);
               }
               if (packed) kind = 1;
            }
            memset(fp->reusable, 0, sizeof(fp->reusable));
            bool simple = inst->Instruction.Opcode != TGSI_OPCODE_TEX;
            for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++)
               simple &= inst->Src[src].Register.File != TGSI_FILE_INPUT;
            if (simple) {
               for (unsigned reg = 0; reg < 24; reg++) {
                  if (reg >= 4 && reg < 16) continue;
                  unsigned needed = 0;
                  for (unsigned t = 0; t < fp->num_temporaries; t++) for (unsigned k = 0; k < 4; k++) {
                     int loc = fp->temp_src[t][k];
                     if (loc < 0 || (loc & 0x7f) != reg) continue;
                     unsigned mask = (loc & TEMP_FX10) ? ((loc & TEMP_HIGH) ? 2 : 1) : 3;
                     bool overwritten = false;
                     for (unsigned d2 = 0; d2 < inst->Instruction.NumDstRegs; d2++) {
                        const struct tgsi_dst_register *out = &inst->Dst[d2].Register;
                        overwritten |= out->File == TGSI_FILE_TEMPORARY && out->Index == t && (out->WriteMask & (1 << k));
                     }
                     if (!overwritten && (live_after[n + 1][t] & (1 << k))) needed |= mask;
                  }
                  for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++) {
                     const struct tgsi_src_register *operand = &inst->Src[src].Register;
                     if (operand->File != TGSI_FILE_TEMPORARY) continue;
                     unsigned swizzle[] = {operand->SwizzleX, operand->SwizzleY, operand->SwizzleZ, operand->SwizzleW};
                     for (unsigned lane = c + 1; lane < 4; lane++) {
                        int loc = fp->temp_src[operand->Index][swizzle[lane]];
                        if ((dst->WriteMask & (1 << lane)) && loc >= 0 && (loc & 0x7f) == reg)
                           needed |= (loc & TEMP_FX10) ? ((loc & TEMP_HIGH) ? 2 : 1) : 3;
                     }
                  }
                  fp->reusable[reg] = (~needed) & 3;
               }
            }
            fp->temp_dst[dst->Index][c] = (live_after[n + 1][dst->Index] & (1 << c)) ?
                                         (fp->texture_rows ? (2 + c / 2) | TEMP_FX10 | (c % 2 ? TEMP_HIGH : 0) :
                                          allocate_scalar(fp, kind)) : 31;
         }
      }
      if (grate_debug & GRATE_DEBUG_TGSI) {
         fprintf(stderr, "ALLOC %u opcode %u", n, inst->Instruction.Opcode);
         for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++) {
            const struct tgsi_src_register *operand = &inst->Src[src].Register;
            if (operand->File == TGSI_FILE_TEMPORARY)
               fprintf(stderr, " src%u[%d,%d,%d,%d]", src, fp->temp_src[operand->Index][0],fp->temp_src[operand->Index][1],fp->temp_src[operand->Index][2],fp->temp_src[operand->Index][3]);
         }
         for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
            const struct tgsi_dst_register *out = &inst->Dst[d].Register;
            if (out->File == TGSI_FILE_TEMPORARY)
               fprintf(stderr, " dst[%d,%d,%d,%d]", fp->temp_dst[out->Index][0],fp->temp_dst[out->Index][1],fp->temp_dst[out->Index][2],fp->temp_dst[out->Index][3]);
         }
         fprintf(stderr, "\n");
      }
      emit_tgsi_instr(fp, inst);
      memset(fp->occupied, 0, sizeof(fp->occupied));
      for (unsigned t = 0; t < fp->num_temporaries; t++) {
         for (unsigned c = 0; c < 4; c++) {
            int reg = fp->temp_dst[t][c];
            if (live_after[n + 1][t] & (1 << c)) {
               assert(reg >= 0 || fp->allocation_failed);
               fp->temp_src[t][c] = reg;
               if (reg >= 0 && reg != 31) {
                  unsigned mask = (reg & TEMP_FX10) ? ((reg & TEMP_HIGH) ? 2 : 1) : 3;
                  fp->occupied[reg & 0x7f] |= mask;
               }
            } else {
               fp->temp_src[t][c] = -1;
            }
         }
      }
      memset(fp->temp_src[fp->num_temporaries], -1, sizeof(fp->temp_src[0]));
   }
   free(live_after);
   util_dynarray_fini(&instructions);

}
