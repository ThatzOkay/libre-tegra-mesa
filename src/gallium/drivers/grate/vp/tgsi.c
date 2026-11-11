#include <assert.h>
#include "grate_compiler.h"
#include "unistd.h"
#include "vpir.h"

#include "tgsi/tgsi_parse.h"

#include "util/u_memory.h"

static struct vp_src_operand
src_undef()
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_UNDEF,
      .index = 0,
      .swizzle = { VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z, VP_SWZ_W }
   };
   return ret;
}

static struct vp_src_operand
attrib(int index, const enum vp_swz swizzle[4], bool negate, bool absolute)
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_ATTRIB,
      .index = index,
      .negate = negate,
      .absolute = absolute
   };
   memcpy(ret.swizzle, swizzle, sizeof(ret.swizzle));
   return ret;
}

static struct vp_src_operand
uniform(int index, const enum vp_swz swizzle[4], bool negate, bool absolute)
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_UNIFORM,
      .index = index,
      .negate = negate,
      .absolute = absolute
   };
   memcpy(ret.swizzle, swizzle, sizeof(ret.swizzle));
   return ret;
}

static struct vp_src_operand
src_temp(int index, const enum vp_swz swizzle[4], bool negate, bool absolute)
{
   struct vp_src_operand ret = {
      .file = VP_SRC_FILE_TEMP,
      .index = index,
      .negate = negate,
      .absolute = absolute
   };
   memcpy(ret.swizzle, swizzle, sizeof(ret.swizzle));
   return ret;
}

static struct vp_dst_operand
dst_undef()
{
   struct vp_dst_operand ret = {
      .file = VP_DST_FILE_UNDEF,
      .index = 0,
      .write_mask = 0,
      .saturate = 0
   };
   return ret;
}

static struct vp_dst_operand
emit_output(struct grate_vp_shader *vp, int index,
            unsigned int write_mask, bool saturate)
{
   vp->output_mask |= 1 << index;
   struct vp_dst_operand ret = {
      .file = VP_DST_FILE_OUTPUT,
      .index = index,
      .write_mask = write_mask,
      .saturate = saturate
   };
   return ret;
}

static struct vp_dst_operand
dst_temp(int index, unsigned int write_mask, bool saturate)
{
   struct vp_dst_operand ret = {
      .file = VP_DST_FILE_TEMP,
      .index = index,
      .write_mask = write_mask,
      .saturate = saturate
   };
   return ret;
}

static struct vp_vec_instr
emit_vec_unop(enum vp_vec_op op, struct vp_dst_operand dst,
              struct vp_src_operand src)
{
   struct vp_vec_instr ret = {
      .op = op,
      .dst = dst,
      .src = { src, src_undef(), src_undef() }
   };
   return ret;
}

static struct vp_vec_instr
emit_vec_binop(enum vp_vec_op op, struct vp_dst_operand dst,
              struct vp_src_operand src0, struct vp_src_operand src1)
{
   struct vp_vec_instr ret = {
      .op = op,
      .dst = dst,
      .src = { src0, src1, src_undef() }
   };
   return ret;
}

static struct vp_vec_instr
emit_vNOP()
{
   struct vp_vec_instr ret = {
      .op = VP_VEC_OP_NOP,
      .dst = dst_undef(),
      .src = { src_undef(), src_undef(), src_undef() }
   };
   return ret;
}

static struct vp_vec_instr
emit_vMOV(struct vp_dst_operand dst, struct vp_src_operand src)
{
   return emit_vec_unop(VP_VEC_OP_MOV, dst, src);
}

static struct vp_vec_instr
emit_vADD(struct vp_dst_operand dst, struct vp_src_operand src0,
          struct vp_src_operand src2)
{
   struct vp_vec_instr ret = {
      .op = VP_VEC_OP_ADD,
      .dst = dst,
      .src = { src0, src_undef(), src2 } // add is "strange" in that it takes src0 and src2
   };
   return ret;
}

#define GEN_V_BINOP(OP) \
static struct vp_vec_instr \
emit_v ## OP (struct vp_dst_operand dst, struct vp_src_operand src0, \
          struct vp_src_operand src1) \
{ \
   return emit_vec_binop(VP_VEC_OP_ ## OP, dst, src0, src1); \
}

GEN_V_BINOP(MUL)
GEN_V_BINOP(DP3)
GEN_V_BINOP(DP4)
GEN_V_BINOP(SLT)
GEN_V_BINOP(MAX)
GEN_V_BINOP(MIN)
GEN_V_BINOP(SNE)
GEN_V_BINOP(SEQ)

static struct vp_vec_instr
emit_vMAD(struct vp_dst_operand dst, struct vp_src_operand src0,
          struct vp_src_operand src1, struct vp_src_operand src2)
{
   struct vp_vec_instr ret = {
      .op = VP_VEC_OP_MAD,
      .dst = dst,
      .src = { src0, src1, src2 }
   };
   return ret;
}

static struct vp_scalar_instr
emit_sNOP()
{
   struct vp_scalar_instr ret = {
      .op = VP_SCALAR_OP_NOP,
      .dst = dst_undef(),
      .src = src_undef()
   };
   return ret;
}

#define GEN_S_UNOP(OP) \
static struct vp_scalar_instr \
emit_s ## OP (struct vp_dst_operand dst, struct vp_src_operand src) \
{ \
   struct vp_scalar_instr ret = { \
      .op = VP_SCALAR_OP_ ## OP, \
      .dst = dst, \
      .src = src \
   }; \
   return ret; \
}

GEN_S_UNOP(RSQ)
GEN_S_UNOP(RCP)

static struct vp_instr *
emit_packed(struct vp_vec_instr vec, struct vp_scalar_instr scalar)
{
   struct vp_instr *ret = CALLOC_STRUCT(vp_instr);
   list_inithead(&ret->link);
   ret->vec = vec;
   ret->scalar = scalar;
   return ret;
}

static struct vp_dst_operand
tgsi_dst_to_vp(struct grate_vp_shader *vp, const struct tgsi_dst_register *dst, bool saturate)
{
   switch (dst->File) {
   case TGSI_FILE_OUTPUT:
      return emit_output(vp, dst->Index, dst->WriteMask, saturate);

   case TGSI_FILE_TEMPORARY:
      return dst_temp(dst->Index, dst->WriteMask, saturate);

   default:
      UNREACHABLE("unsupported output");
   }
}

static struct vp_src_operand
tgsi_src_to_vp(struct grate_vp_shader *vp, const struct tgsi_src_register *src)
{
   enum vp_swz swizzle[4] = {
      src->SwizzleX,
      src->SwizzleY,
      src->SwizzleZ,
      src->SwizzleW
   };
   bool negate = src->Negate != 0;
   bool absolute = src->Absolute != 0;

   switch (src->File) {
   case TGSI_FILE_INPUT:
      return attrib(src->Index, swizzle, negate, absolute);

   case TGSI_FILE_CONSTANT:
      /* the application's uniforms follow the immediates, see
       * grate_compiler.h */
      return uniform(vp->num_immediates + src->Index, swizzle, negate,
                     absolute);

   case TGSI_FILE_TEMPORARY:
      return src_temp(src->Index, swizzle, negate, absolute);

   case TGSI_FILE_IMMEDIATE:
      /* the values are recorded in grate_tgsi_to_vp() and uploaded to the
       * first slots, see grate_compiler.h */
      assert(src->Index >= 0 && (unsigned)src->Index < vp->num_immediates);
      return uniform(src->Index, swizzle, negate, absolute);

   default:
      UNREACHABLE("unsupported input!");
   }
}

static bool
legalize_fetches(struct grate_vp_shader *vp, struct vp_instr *instr,
                 struct list_head *out);

static struct vp_instr *
tgsi_to_vp(struct grate_vp_shader *vp, const struct tgsi_full_instruction *inst)
{
   bool saturate = inst->Instruction.Saturate != 0;
   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_MOV:
      return emit_packed(emit_vMOV(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_ADD:
      return emit_packed(emit_vADD(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_MUL:
      return emit_packed(emit_vMUL(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_DP3:
      return emit_packed(emit_vDP3(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_DP4:
      return emit_packed(emit_vDP4(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_SLT:
      return emit_packed(emit_vSLT(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_MAX:
      return emit_packed(emit_vMAX(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_MIN:
      return emit_packed(
         emit_vMIN(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
         emit_sNOP());

   case TGSI_OPCODE_SNE:
      return emit_packed(
         emit_vSNE(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
         emit_sNOP());

   case TGSI_OPCODE_SEQ:
      return emit_packed(
         emit_vSEQ(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
         emit_sNOP());

   case TGSI_OPCODE_SQRT: {
      /* The scalar unit has no square root. NVIDIA's compiler computes
       * sqrt(x) = RCP(RSQ(x)), which keeps sqrt(0) = RCP(inf) = 0, where
       * x * RSQ(x) would give 0 * inf = NaN. */
      static const enum vp_swz xxxx[4] = {VP_SWZ_X, VP_SWZ_X, VP_SWZ_X,
                                          VP_SWZ_X};
      if (vp->num_temps >= 62) {
         fprintf(stderr, "GRATE VERTEX: out of temporaries for SQRT\n");
         vp->failed = true;
         return 0;
      }
      unsigned t = vp->num_temps++;
      struct vp_instr *rsq = emit_packed(
         emit_vNOP(), emit_sRSQ(dst_temp(t, 0xf, false),
                                tgsi_src_to_vp(vp, &inst->Src[0].Register)));
      legalize_fetches(vp, rsq, &vp->instructions);
      list_addtail(&rsq->link, &vp->instructions);
      return emit_packed(
         emit_vNOP(),
         emit_sRCP(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                   src_temp(t, xxxx, false, false)));
   }

   case TGSI_OPCODE_DP2: {
      /* The vector unit has no two-component dot product. t = a * b, then
       * dst = t.x + t.y in every lane (TGSI replicates the result). */
      static const enum vp_swz xxxx[4] = {VP_SWZ_X, VP_SWZ_X, VP_SWZ_X,
                                          VP_SWZ_X};
      static const enum vp_swz yyyy[4] = {VP_SWZ_Y, VP_SWZ_Y, VP_SWZ_Y,
                                          VP_SWZ_Y};
      if (vp->num_temps >= 62) {
         fprintf(stderr, "GRATE VERTEX: out of temporaries for DP2\n");
         vp->failed = true;
         return 0;
      }
      unsigned t = vp->num_temps++;
      struct vp_instr *mul =
         emit_packed(emit_vMUL(dst_temp(t, 0xf, false),
                               tgsi_src_to_vp(vp, &inst->Src[0].Register),
                               tgsi_src_to_vp(vp, &inst->Src[1].Register)),
                     emit_sNOP());
      legalize_fetches(vp, mul, &vp->instructions);
      list_addtail(&mul->link, &vp->instructions);
      return emit_packed(
         emit_vADD(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                   src_temp(t, xxxx, false, false),
                   src_temp(t, yyyy, false, false)),
         emit_sNOP());
   }

   case TGSI_OPCODE_CMP: {
      /* dst = src0 < 0 ? src1 : src2, laid out like NVIDIA's compiler does
       * a select: set the condition flags from src0 (the instruction
       * writes no register: r63), write src2, then write src1 on the
       * lanes whose flag says less than zero. Exact, unlike a blend, and
       * it needs no zero constant. */
      static const enum vp_swz identity[4] = {VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z,
                                              VP_SWZ_W};
      const struct tgsi_dst_register *d = &inst->Dst[0].Register;
      unsigned t = 0;
      bool scratch = false;
      /* a destination that is also a source would be overwritten before
       * the predicated write reads it */
      if (d->File == TGSI_FILE_TEMPORARY &&
          ((inst->Src[1].Register.File == TGSI_FILE_TEMPORARY &&
            inst->Src[1].Register.Index == d->Index) ||
           (inst->Src[2].Register.File == TGSI_FILE_TEMPORARY &&
            inst->Src[2].Register.Index == d->Index))) {
         if (vp->num_temps >= 62) {
            fprintf(stderr, "GRATE VERTEX: out of temporaries for CMP\n");
            vp->failed = true;
            return 0;
         }
         t = vp->num_temps++;
         scratch = true;
      }
      struct vp_dst_operand dst = scratch ? dst_temp(t, d->WriteMask, false)
                                          : tgsi_dst_to_vp(vp, d, saturate);

      struct vp_instr *flags =
         emit_packed(emit_vMOV(dst_temp(63, d->WriteMask, false),
                               tgsi_src_to_vp(vp, &inst->Src[0].Register)),
                     emit_sNOP());
      flags->cond.set = true;
      legalize_fetches(vp, flags, &vp->instructions);
      list_addtail(&flags->link, &vp->instructions);

      struct vp_instr *no = emit_packed(
         emit_vMOV(dst, tgsi_src_to_vp(vp, &inst->Src[2].Register)),
         emit_sNOP());
      legalize_fetches(vp, no, &vp->instructions);
      list_addtail(&no->link, &vp->instructions);

      struct vp_instr *yes = emit_packed(
         emit_vMOV(dst, tgsi_src_to_vp(vp, &inst->Src[1].Register)),
         emit_sNOP());
      yes->cond.check = true;
      yes->cond.lt = true;
      memcpy(yes->cond.swizzle, identity, sizeof(identity));
      if (!scratch)
         return yes;
      legalize_fetches(vp, yes, &vp->instructions);
      list_addtail(&yes->link, &vp->instructions);

      return emit_packed(emit_vMOV(tgsi_dst_to_vp(vp, d, saturate),
                                   src_temp(t, identity, false, false)),
                         emit_sNOP());
   }

   case TGSI_OPCODE_MAD:
      return emit_packed(emit_vMAD(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[1].Register),
                                   tgsi_src_to_vp(vp, &inst->Src[2].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_LRP: {
      /* LRP(a, b, c) = a * b + (1 - a) * c, as two multiply-adds around a
       * temporary t = c - a * c. The first goes out here, the caller adds the
       * second. t gets a temporary of its own, which keeps the scratch
       * temporaries of legalize_fetches() off it. */
      static const enum vp_swz identity[4] = {VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z,
                                              VP_SWZ_W};
      if (vp->num_temps >= 62) {
         fprintf(stderr, "GRATE VERTEX: out of temporaries for LRP\n");
         vp->failed = true;
         return 0;
      }
      unsigned t = vp->num_temps++;
      struct vp_src_operand a = tgsi_src_to_vp(vp, &inst->Src[0].Register);
      struct vp_src_operand minus_a = a;
      minus_a.negate = !a.negate;
      struct vp_src_operand c = tgsi_src_to_vp(vp, &inst->Src[2].Register);
      struct vp_instr *first =
         emit_packed(emit_vMAD(dst_temp(t, 0xf, false), minus_a, c, c),
                     emit_sNOP());
      legalize_fetches(vp, first, &vp->instructions);
      list_addtail(&first->link, &vp->instructions);
      return emit_packed(
         emit_vMAD(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate), a,
                   tgsi_src_to_vp(vp, &inst->Src[1].Register),
                   src_temp(t, identity, false, false)),
         emit_sNOP());
   }

   case TGSI_OPCODE_RSQ:
      return emit_packed(emit_vNOP(),
                         emit_sRSQ(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)));
      
   case TGSI_OPCODE_RCP:
      return emit_packed(emit_vNOP(),
                           emit_sRCP(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                    tgsi_src_to_vp(vp, &inst->Src[0].Register)));

   default:
      /* Not a bug of the application: this compiler knows a small part of
       * TGSI, and real programs use more. Refuse the shader (no shader is
       * created, the draws that need it are skipped, see grate_draw_vbo())
       * instead of aborting the whole process, which took a browser's
       * compositor down for one missing opcode. The message names the
       * opcode to implement next. The instruction is dropped from the
       * program, which is discarded anyway. */
      fprintf(stderr, "GRATE VERTEX TGSI UNIMPLEMENTED: 0x%02x\n",
              inst->Instruction.Opcode);
      vp->failed = true;
      return 0;
   }
}

/* A packed instruction has ONE attribute-fetch index and ONE uniform-fetch
 * index shared by all its sources (see grate_vp_pack()). When an instruction
 * reads a second, different attribute or uniform, copy that operand into a
 * scratch temporary with a preceding MOV and read the temporary instead.
 */
static bool
legalize_fetches(struct grate_vp_shader *vp, struct vp_instr *instr,
                 struct list_head *out)
{
   struct vp_src_operand *srcs[4] = {&instr->vec.src[0], &instr->vec.src[1],
                                     &instr->vec.src[2], &instr->scalar.src};
   int attr_fetch = -1, uniform_fetch = -1;
   unsigned next_temp = vp->num_temps;
   static const enum vp_swz identity[4] = {VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z,
                                           VP_SWZ_W};

   for (unsigned i = 0; i < 4; ++i) {
      struct vp_src_operand *src = srcs[i];
      int *fetch;

      if (src->file == VP_SRC_FILE_ATTRIB)
         fetch = &attr_fetch;
      else if (src->file == VP_SRC_FILE_UNIFORM)
         fetch = &uniform_fetch;
      else
         continue;

      if (*fetch < 0 || *fetch == src->index) {
         *fetch = src->index;
         continue;
      }

      /* the hardware has 63 usable temporaries (index 63 disables the write) */
      if (next_temp >= 63) {
         fprintf(stderr, "GRATE VERTEX: out of temporaries while splitting "
                         "operand fetches\n");
         return false;
      }

      struct vp_src_operand copy = *src;
      memcpy(copy.swizzle, identity, sizeof(copy.swizzle));
      copy.negate = false;
      copy.absolute = false;

      struct vp_instr *mov = emit_packed(
         emit_vMOV(dst_temp(next_temp, 0xf, false), copy), emit_sNOP());
      list_addtail(&mov->link, out);

      /* the consumer keeps the original swizzle, negate and absolute */
      src->file = VP_SRC_FILE_TEMP;
      src->index = next_temp++;
   }
   return true;
}

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi)
{
   list_inithead(&vp->instructions);
   vp->output_mask = 0;
   vp->num_immediates = 0;
   vp->num_temps = 0;
   vp->failed = false;

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_IMMEDIATE: {
         const struct tgsi_full_immediate *imm =
            &tgsi->FullToken.FullImmediate;
         unsigned n =
            imm->Immediate.NrTokens - 1; /* number of 32-bit values: 1 to 4 */

         assert(vp->num_immediates < GRATE_VP_MAX_IMMEDIATES);
         for (unsigned i = 0; i < 4; ++i)
            vp->immediates[vp->num_immediates][i] =
               i < n ? imm->u[i].Uint : 0;
         vp->num_immediates++;
         break;
      }

      case TGSI_TOKEN_TYPE_DECLARATION: {
         const struct tgsi_full_declaration *decl =
            &tgsi->FullToken.FullDeclaration;
         if (decl->Declaration.File == TGSI_FILE_TEMPORARY &&
             decl->Range.Last + 1 > vp->num_temps)
            vp->num_temps = decl->Range.Last + 1;
         break;
      }

      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode != TGSI_OPCODE_END) {
            struct vp_instr *instr = tgsi_to_vp(vp, &tgsi->FullToken.FullInstruction);
            if (!instr) 
               continue;
            if (!legalize_fetches(vp, instr, &vp->instructions))
               vp->failed = true;
            list_addtail(&instr->link, &vp->instructions);
         }
         break;
      }
   }
}
