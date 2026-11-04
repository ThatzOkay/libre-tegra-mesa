#include "grate_common.h"
#include "grate_compiler.h"
#include "fpir.h"

#include "tgsi/tgsi_parse.h"

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
fp_alu_src_reg(int index)
{
   assert(index >= 0 && index < 8);
   struct fp_alu_src_operand src = {
      .index = 16 + index
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

static struct fp_alu_dst_operand
fp_alu_dst(const struct tgsi_dst_register *dst, int subreg, bool saturate)
{
   struct fp_alu_dst_operand ret = { 0 };

   ret.index = 16 + dst->Index * 4;
   if (dst->File == TGSI_FILE_OUTPUT) {
      ret.index = 2; // HACK: r2+r3 to match hard-coded store shader for now

      // fixed10
      // swizzle RGBA -> BGRA
      int o = subreg < 3 ? (2 - subreg) : 3;
      ret.index += o / 2;
      ret.write_low_sub_reg = (o % 2) == 0;
      ret.write_high_sub_reg = (o % 2) != 0;
   } else {
      assert(dst->File == TGSI_FILE_TEMPORARY && dst->Index < 2);
      ret.index += subreg;
      ret.write_low_sub_reg = true;
      ret.write_high_sub_reg = true;
   }

   ret.saturate = saturate;

   return ret;
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
      if (imm != 0.0f && imm != 1.0f)
         fprintf(stderr, "GRATE FRAG: immediate %f is not supported (only 0.0 and 1.0), using 0.0\n", imm);
      value = imm == 1.0f ? fp_alu_src_one() : fp_alu_src_zero();
      break;
   }
   case TGSI_FILE_CONSTANT:
      fprintf(stderr, "GRATE FRAG: constants (uniforms) are not supported in fragment shaders yet, using 0.0\n");
      value = fp_alu_src_zero();
      break;
   case TGSI_FILE_TEMPORARY:
      value = fp_alu_src_reg(src->Index * 4 + comp);
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
          const struct tgsi_src_register *rhs)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);

   struct fp_mfu_instr *mfu = NULL;
   if (src->File == TGSI_FILE_INPUT || (rhs && rhs->File == TGSI_FILE_INPUT)) {
      mfu = CALLOC_STRUCT(fp_mfu_instr);
      list_inithead(&mfu->link);
      setup_interpolation(mfu);
   }

   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   int alu_instrs = 0;
   list_inithead(&alu->link);
   for (int i = 0; i < 4; ++i) {
      if ((dst->WriteMask & (1 << i)) == 0)
         continue;

      struct fp_alu_src_operand a = fp_tgsi_src(fp, src, i, mfu);
      struct fp_alu_src_operand b = rhs ? fp_tgsi_src(fp, rhs, i, mfu) : fp_alu_src_one();
      struct fp_alu_instr op = fp_alu_sMOV(fp_alu_dst(dst, i, saturate), a);
      op.src[1] = b; /* MAD(a, b, zero, one) implements MOV or MUL. */
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
   if (!mfu && !list_is_empty(&fp->fp_instructions)) {
      struct fp_instr *previous = list_last_entry(&fp->fp_instructions, struct fp_instr, link);
      if (!previous->dw.enable && previous->alu_sched.num_instructions < 3) {
         if (previous->alu_sched.num_instructions == 1 &&
             (src->File == TGSI_FILE_TEMPORARY || (rhs && rhs->File == TGSI_FILE_TEMPORARY))) {
            struct fp_alu_instr_packet *nop = CALLOC_STRUCT(fp_alu_instr_packet);
            list_inithead(&nop->link);
            /* Allow the GPR result to become available before it is read. */
            struct fp_alu_dst_operand discard = {.index = 31};
            for (unsigned n = 0; n < 4; n++)
               nop->slots[n] = fp_alu_sMOV(discard, fp_alu_src_zero());
            list_addtail(&nop->link, &alu->link);
            previous->alu_sched.num_instructions++;
         }
         previous->alu_sched.num_instructions++;
         previous->dw = inst->dw;
         FREE(inst);
         return;
      }
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
      alu->slots[slot++] = fp_alu_sMOV(fp_alu_dst(dst, i, tgsi->Instruction.Saturate), src);
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

   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_TEX:
      emit_TEX(fp, inst);
      break;
   case TGSI_OPCODE_MOV:
      emit_vALU(fp, &inst->Dst[0].Register, saturate,
                    &inst->Src[0].Register, NULL);
      break;

   case TGSI_OPCODE_MUL: {
      const struct tgsi_src_register *a = &inst->Src[0].Register;
      const struct tgsi_src_register *b = &inst->Src[1].Register;
      struct tgsi_src_register staged = {0};
      if (a->File == TGSI_FILE_INPUT && b->File == TGSI_FILE_INPUT && a->Index != b->Index) {
         /* One MFU packet can interpolate one vec4. Preserve A in an unused general-purpose vec4
          * before loading B. Scratch registers must not alias a live
          * TGSI temporary. General register allocation is still missing.
          */
         assert(fp->num_temporaries <= 1);
         struct tgsi_dst_register scratch = {0};
         scratch.File = TGSI_FILE_TEMPORARY;
         scratch.Index = fp->num_temporaries;
         scratch.WriteMask = TGSI_WRITEMASK_XYZW;
         emit_vALU(fp, &scratch, false, a, NULL);
         staged.File = TGSI_FILE_TEMPORARY;
         staged.Index = fp->num_temporaries;
         staged.SwizzleX = TGSI_SWIZZLE_X;
         staged.SwizzleY = TGSI_SWIZZLE_Y;
         staged.SwizzleZ = TGSI_SWIZZLE_Z;
         staged.SwizzleW = TGSI_SWIZZLE_W;
         a = &staged;
      }
      emit_vALU(fp, &inst->Dst[0].Register, saturate, a, b);

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
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;
   fp->num_immediates = 0;
   fp->num_temporaries = 0;

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
            emit_tgsi_instr(fp, &tgsi->FullToken.FullInstruction);
         break;
      }
   }

}
