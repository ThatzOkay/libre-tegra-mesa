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
emit_vFLR(struct vp_dst_operand dst, struct vp_src_operand src)
{
   return emit_vec_unop(VP_VEC_OP_FLR, dst, src);
}

static struct vp_vec_instr
emit_vFRC(struct vp_dst_operand dst, struct vp_src_operand src)
{
   return emit_vec_unop(VP_VEC_OP_FRC, dst, src);
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
GEN_V_BINOP(SGE)
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
GEN_S_UNOP(SIN)
GEN_S_UNOP(COS)
GEN_S_UNOP(LG2)
GEN_S_UNOP(EX2)

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

   case TGSI_FILE_CONSTANT: {
      /* the application's uniforms follow the immediates, see
       * grate_compiler.h */
      struct vp_src_operand ret =
         uniform(vp->num_immediates + src->Index, swizzle, negate, absolute);
      /* CONST[ADDR[0].x + index], checked in tgsi_to_vp() */
      ret.relative = src->Indirect != 0;
      return ret;
   }

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

   /* the only indirect addressing there is: CONST[ADDR[0].x + index] */
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      const struct tgsi_full_src_register *src = &inst->Src[i];
      if (src->Register.Indirect &&
          (src->Register.File != TGSI_FILE_CONSTANT ||
           src->Indirect.File != TGSI_FILE_ADDRESS ||
           src->Indirect.Index != 0 ||
           src->Indirect.Swizzle != TGSI_SWIZZLE_X)) {
         fprintf(stderr, "GRATE VERTEX: unsupported indirect addressing\n");
         vp->failed = true;
         return 0;
      }
   }

   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_ARL: {
      /* A0.x = floor(src.x): the vector op ARL, as in NVIDIA's compiler
       * (idx-attr in tools/gpu/grate-vertex-ops-test.c). Like NVIDIA's
       * compiler it writes the register it reads (ARL r0.y A=r0.yyyy): with
       * another destination, even r63 or a scratch, the address register was
       * not loaded on the tablet (idx-attr, idx-two read slot 0). The floor
       * stays in that register, which changes nothing, because an integer
       * is a float holding an integer here. A source that is not lane x of
       * a temporary, or that has a modifier, is copied to one first. */
      static const enum vp_swz xxxx[4] = {VP_SWZ_X, VP_SWZ_X, VP_SWZ_X,
                                          VP_SWZ_X};
      const struct tgsi_src_register *src = &inst->Src[0].Register;
      if (inst->Dst[0].Register.File != TGSI_FILE_ADDRESS ||
          inst->Dst[0].Register.Index != 0 ||
          !(inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_X)) {
         fprintf(stderr, "GRATE VERTEX: unsupported address register\n");
         vp->failed = true;
         return 0;
      }

      unsigned t = src->Index;
      struct vp_src_operand a = tgsi_src_to_vp(vp, src);
      if (src->File != TGSI_FILE_TEMPORARY ||
          src->SwizzleX != TGSI_SWIZZLE_X || src->Negate || src->Absolute) {
         if (vp->num_temps >= 62) {
            fprintf(stderr, "GRATE VERTEX: out of temporaries for ARL\n");
            vp->failed = true;
            return 0;
         }
         t = vp->num_temps++;
         struct vp_instr *copy =
            emit_packed(emit_vMOV(dst_temp(t, 0x1, false), a), emit_sNOP());
         legalize_fetches(vp, copy, &vp->instructions);
         list_addtail(&copy->link, &vp->instructions);
         a = src_temp(t, xxxx, false, false);
      }
      return emit_packed(
         emit_vec_unop(VP_VEC_OP_ARL, dst_temp(t, 0x1, false), a),
         emit_sNOP());
   }

   case TGSI_OPCODE_TRUNC: {
      /* trunc(x) = x < 0 ? -floor(|x|) : floor(|x|), the int() of
       * NVIDIA's compiler (idx-offset): the flags from x, FLR of |x|, then
       * the lanes whose flag says negative are negated. The flags come
       * first, the destination may be x itself. */
      static const enum vp_swz identity[4] = {VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z,
                                              VP_SWZ_W};
      const struct tgsi_dst_register *d = &inst->Dst[0].Register;
      bool scratch = d->File != TGSI_FILE_TEMPORARY;
      unsigned t = d->Index;
      if (saturate) {
         fprintf(stderr, "GRATE VERTEX: saturated TRUNC\n");
         vp->failed = true;
         return 0;
      }
      if (scratch) {
         if (vp->num_temps >= 62) {
            fprintf(stderr, "GRATE VERTEX: out of temporaries for TRUNC\n");
            vp->failed = true;
            return 0;
         }
         t = vp->num_temps++;
      }

      struct vp_instr *flags =
         emit_packed(emit_vMOV(dst_temp(63, d->WriteMask, false),
                               tgsi_src_to_vp(vp, &inst->Src[0].Register)),
                     emit_sNOP());
      flags->cond.set = true;
      legalize_fetches(vp, flags, &vp->instructions);
      list_addtail(&flags->link, &vp->instructions);

      struct vp_src_operand x = tgsi_src_to_vp(vp, &inst->Src[0].Register);
      x.negate = false;
      x.absolute = true;
      struct vp_instr *flr = emit_packed(
         emit_vFLR(dst_temp(t, d->WriteMask, false), x), emit_sNOP());
      legalize_fetches(vp, flr, &vp->instructions);
      list_addtail(&flr->link, &vp->instructions);

      struct vp_instr *neg =
         emit_packed(emit_vMOV(dst_temp(t, d->WriteMask, false),
                               src_temp(t, identity, true, false)),
                     emit_sNOP());
      neg->cond.check = true;
      neg->cond.lt = true;
      memcpy(neg->cond.swizzle, identity, sizeof(identity));
      if (!scratch)
         return neg;
      list_addtail(&neg->link, &vp->instructions);

      return emit_packed(emit_vMOV(tgsi_dst_to_vp(vp, d, false),
                                   src_temp(t, identity, false, false)),
                         emit_sNOP());
   }

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

   case TGSI_OPCODE_FLR:
      /* One vector op like MOV: NVIDIA's compiler compiles floor() to a
       * FLR with the operand in slot A (decoded with ar20vp-dump.py). */
      return emit_packed(emit_vFLR(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_FRC:
      /* One vector op like FLR, operand in slot A (NVIDIA's compiler
       * compiles fract() to a FRC, decoded with ar20vp-dump.py). */
      return emit_packed(emit_vFRC(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)),
                         emit_sNOP());

   case TGSI_OPCODE_SGE:
      /* a >= b with the operands in slots A and B, like NVIDIA's compiler
       * for step() and greaterThanEqual(). */
      return emit_packed(
         emit_vSGE(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
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

   case TGSI_OPCODE_SIN:
      /* The scalar unit has SIN and takes the angle in radians as it is,
       * without range reduction, like NVIDIA's compiler (sin.vert). */
      return emit_packed(emit_vNOP(),
                         emit_sSIN(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)));

   case TGSI_OPCODE_COS:
      /* Like SIN (cos.vert). */
      return emit_packed(emit_vNOP(),
                         emit_sCOS(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)));

   case TGSI_OPCODE_EX2:
      /* The scalar unit has EX2 and LG2, exp2() and log2() are one each in
       * NVIDIA's compiler (exp2.vert). */
      return emit_packed(emit_vNOP(),
                         emit_sEX2(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)));

   case TGSI_OPCODE_LG2:
      return emit_packed(emit_vNOP(),
                         emit_sLG2(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                                   tgsi_src_to_vp(vp, &inst->Src[0].Register)));

   case TGSI_OPCODE_POW: {
      /* pow(x, y) = EX2(y * LG2(x)), the same three steps as NVIDIA's
       * compiler (pow.vert): LG2 on the scalar unit, a vector MUL of the
       * lane x, then EX2. */
      static const enum vp_swz xxxx[4] = {VP_SWZ_X, VP_SWZ_X, VP_SWZ_X,
                                          VP_SWZ_X};
      if (vp->num_temps >= 62) {
         fprintf(stderr, "GRATE VERTEX: out of temporaries for POW\n");
         vp->failed = true;
         return 0;
      }
      unsigned t = vp->num_temps++;
      struct vp_instr *lg2 = emit_packed(
         emit_vNOP(), emit_sLG2(dst_temp(t, 0x1, false),
                                tgsi_src_to_vp(vp, &inst->Src[0].Register)));
      legalize_fetches(vp, lg2, &vp->instructions);
      list_addtail(&lg2->link, &vp->instructions);
      struct vp_instr *mul = emit_packed(
         emit_vMUL(dst_temp(t, 0x1, false), src_temp(t, xxxx, false, false),
                   tgsi_src_to_vp(vp, &inst->Src[1].Register)),
         emit_sNOP());
      legalize_fetches(vp, mul, &vp->instructions);
      list_addtail(&mul->link, &vp->instructions);
      return emit_packed(
         emit_vNOP(),
         emit_sEX2(tgsi_dst_to_vp(vp, &inst->Dst[0].Register, saturate),
                   src_temp(t, xxxx, false, false)));
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
   bool uniform_relative = false;
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

      /* the relative addressing bit is one per instruction */
      if (*fetch < 0 ||
          (*fetch == src->index &&
           (fetch == &attr_fetch || uniform_relative == src->relative))) {
         *fetch = src->index;
         if (fetch == &uniform_fetch)
            uniform_relative = src->relative;
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

/* Control flow. NVIDIA's compiler turns an if without a loop into
 * predicated writes and keeps a BRA for loops only (see
 * tools/gpu/ar20vp-dump.py on cgtest output). NIR already flattens the small
 * ifs, so what reaches this point are loops, with an IF around a BRK, and ifs
 * that contain a loop. They are translated one to one into BRA, whose target
 * is a label until the program is complete (resolve_labels()):
 *
 *    IF c         flags = c; BRA if flag == 0 to ELSE (or ENDIF)
 *    ELSE         BRA to ENDIF, define the label of the else part
 *    ENDIF        define the label
 *    BGNLOOP      define the label of the top
 *    BRK, CONT    BRA to the end of the loop, to its top
 *    ENDLOOP      BRA to the top, define the label of the end
 *
 * The flags are set by the instruction right before the BRA that uses them,
 * like NVIDIA's compiler does, so nothing else can change them in between. */
static void
emit_label(struct grate_vp_shader *vp, unsigned label)
{
   struct vp_instr *instr = emit_packed(emit_vNOP(), emit_sNOP());
   instr->is_label = true;
   instr->label = label;
   list_addtail(&instr->link, &vp->instructions);
}

static void
emit_branch(struct grate_vp_shader *vp, unsigned label, bool lt, bool eq,
            bool gt, enum vp_swz lane)
{
   struct vp_vec_instr vec = emit_vNOP();
   struct vp_scalar_instr scalar = emit_sNOP();

   /* no register is written: r63 */
   vec.dst = dst_temp(63, 0, false);
   scalar.op = VP_SCALAR_OP_BRA;
   scalar.dst = dst_temp(63, 0, false);

   struct vp_instr *instr = emit_packed(vec, scalar);
   instr->branch.label = label;
   instr->branch.lt = lt;
   instr->branch.eq = eq;
   instr->branch.gt = gt;
   instr->branch.lane = lane;
   list_addtail(&instr->link, &vp->instructions);
}

static struct grate_vp_cf *
cf_push(struct grate_vp_shader *vp, bool is_loop)
{
   if (vp->cf_depth >= GRATE_VP_MAX_CF_DEPTH) {
      fprintf(stderr, "GRATE VERTEX: control flow nested too deep\n");
      vp->failed = true;
      return NULL;
   }
   struct grate_vp_cf *cf = &vp->cf[vp->cf_depth++];
   memset(cf, 0, sizeof(*cf));
   cf->is_loop = is_loop;
   cf->label_a = vp->num_labels++;
   if (is_loop)
      cf->label_b = vp->num_labels++;
   return cf;
}

/* The innermost open loop, for BRK and CONT. */
static struct grate_vp_cf *
cf_loop(struct grate_vp_shader *vp)
{
   for (unsigned i = vp->cf_depth; i > 0; i--)
      if (vp->cf[i - 1].is_loop)
         return &vp->cf[i - 1];
   fprintf(stderr, "GRATE VERTEX: BRK or CONT outside of a loop\n");
   vp->failed = true;
   return NULL;
}

/* The innermost open construct, which has to be a loop or an IF. */
static struct grate_vp_cf *
cf_top(struct grate_vp_shader *vp, bool is_loop)
{
   if (!vp->cf_depth || vp->cf[vp->cf_depth - 1].is_loop != is_loop) {
      fprintf(stderr, "GRATE VERTEX: unbalanced control flow\n");
      vp->failed = true;
      return NULL;
   }
   return &vp->cf[vp->cf_depth - 1];
}

/* Can the flags be taken from `prev` instead of a MOV of the condition? When
 * prev is a compare that wrote the lane of the temporary the IF reads: the
 * flags then say whether that result is zero. */
static bool
flags_from_compare(const struct vp_instr *prev,
                   const struct tgsi_src_register *cond, enum vp_swz lane)
{
   if (prev->scalar.op != VP_SCALAR_OP_NOP || prev->cond.check ||
       prev->cond.set || prev->vec.dst.file != VP_DST_FILE_TEMP ||
       cond->File != TGSI_FILE_TEMPORARY ||
       prev->vec.dst.index != cond->Index ||
       !(prev->vec.dst.write_mask & (1u << lane)))
      return false;

   switch (prev->vec.op) {
   case VP_VEC_OP_SLT:
   case VP_VEC_OP_SGE:
   case VP_VEC_OP_SEQ:
   case VP_VEC_OP_SNE:
      return true;
   default:
      return false;
   }
}

/* IF c; BRK; ENDIF (and the same with CONT) is flags = c, BRA if flag == 0
 * over a BRA that always jumps. One BRA that jumps when the flag is not
 * zero does the same, which is what NVIDIA's compiler emits for a break
 * (loop-break-first in tools/gpu/grate-vertex-ops-test.c). The flags are
 * still set by the instruction right before it. */
static bool
fold_jump_in_if(struct grate_vp_shader *vp, const struct grate_vp_cf *cf)
{
   if (list_is_empty(&vp->instructions))
      return false;

   struct vp_instr *jump =
      list_last_entry(&vp->instructions, struct vp_instr, link);
   if (jump->is_label || jump->scalar.op != VP_SCALAR_OP_BRA ||
       !jump->branch.lt || !jump->branch.eq || !jump->branch.gt ||
       jump->link.prev == &vp->instructions)
      return false;

   struct vp_instr *skip = list_entry(jump->link.prev, struct vp_instr, link);
   if (skip->is_label || skip->scalar.op != VP_SCALAR_OP_BRA ||
       skip->branch.label != cf->label_a || skip->branch.lt ||
       !skip->branch.eq || skip->branch.gt)
      return false;

   jump->branch.lt = true;
   jump->branch.eq = false;
   jump->branch.gt = true;
   jump->branch.lane = skip->branch.lane;
   list_del(&skip->link);
   FREE(skip);
   return true;
}

/* NVIDIA's compiler rotates a loop: the test that NIR puts at the top
 * (header; IF c; BRK; ENDIF) goes first once as a guard and then again at the
 * bottom as the only branch of an iteration (dloop.vert, lbound.vert):
 *
 *    L: header; if (c) goto END; body; goto L
 *    -->  header; if (c) goto END; TOP: body; L: header; if (!c) goto TOP;
 * END:
 *
 * (CONT jumps to L.) That saves one jump per iteration. It needs a header of
 * a few instructions without branches that ends with the flags of the break,
 * which fold_jump_in_if() has made out of the IF. */
static bool
rotate_loop(struct grate_vp_shader *vp, const struct grate_vp_cf *cf)
{
   struct vp_instr *top = NULL;

   list_for_each_entry_rev (struct vp_instr, instr, &vp->instructions, link) {
      if (instr->is_label && instr->label == cf->label_a) {
         top = instr;
         break;
      }
   }
   if (!top)
      return false;

   struct list_head *first = top->link.next;
   struct vp_instr *brk = NULL;
   unsigned header = 0;
   for (struct list_head *l = first; l != &vp->instructions; l = l->next) {
      struct vp_instr *instr = list_entry(l, struct vp_instr, link);
      if (instr->is_label)
         return false;
      if (instr->scalar.op == VP_SCALAR_OP_BRA) {
         brk = instr;
         break;
      }
      if (++header > 4)
         return false;
   }
   if (!brk || !header || brk->branch.label != cf->label_b ||
       !brk->branch.lt || brk->branch.eq || !brk->branch.gt ||
       !list_entry(brk->link.prev, struct vp_instr, link)->cond.set)
      return false;

   unsigned label_body = vp->num_labels++;
   struct vp_instr *body = emit_packed(emit_vNOP(), emit_sNOP());
   body->is_label = true;
   body->label = label_body;
   list_add(&body->link, &brk->link);

   /* CONT jumps to the test at the bottom now */
   list_del(&top->link);
   list_addtail(&top->link, &vp->instructions);

   for (struct list_head *l = first; l != &brk->link; l = l->next) {
      struct vp_instr *copy = MALLOC(sizeof(*copy));
      if (!copy) {
         vp->failed = true;
         return true;
      }
      *copy = *list_entry(l, struct vp_instr, link);
      list_addtail(&copy->link, &vp->instructions);
   }
   emit_branch(vp, label_body, false, true, false, brk->branch.lane);
   return true;
}

/* Returns false when the opcode is not control flow. */
static bool
tgsi_control_flow(struct grate_vp_shader *vp,
                  const struct tgsi_full_instruction *inst)
{
   struct grate_vp_cf *cf;

   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_IF: {
      cf = cf_push(vp, false);
      if (!cf)
         return true;
      const struct tgsi_src_register *cond = &inst->Src[0].Register;
      /* the lane of the flags that holds the condition (a float, nonzero is
       * true) */
      enum vp_swz lane = VP_SWZ_X;
      struct vp_instr *prev = NULL;
      if (!list_is_empty(&vp->instructions))
         prev = list_last_entry(&vp->instructions, struct vp_instr, link);
      if (prev && !prev->is_label &&
          flags_from_compare(prev, cond, cond->SwizzleX)) {
         /* the compare that made the condition sets the flags itself, as in
          * NVIDIA's compiler (SLT r63.x with the flags set); they are those
          * of the lane the condition is read from */
         prev->cond.set = true;
         lane = cond->SwizzleX;
      } else {
         struct vp_instr *flags = emit_packed(
            emit_vMOV(dst_temp(63, 0x1, false), tgsi_src_to_vp(vp, cond)),
            emit_sNOP());
         flags->cond.set = true;
         if (!legalize_fetches(vp, flags, &vp->instructions))
            vp->failed = true;
         list_addtail(&flags->link, &vp->instructions);
      }
      emit_branch(vp, cf->label_a, false, true, false, lane);
      return true;
   }

   case TGSI_OPCODE_ELSE:
      cf = cf_top(vp, false);
      if (!cf)
         return true;
      cf->has_else = true;
      cf->label_b = vp->num_labels++;
      emit_branch(vp, cf->label_b, true, true, true, VP_SWZ_X);
      emit_label(vp, cf->label_a);
      return true;

   case TGSI_OPCODE_ENDIF:
      cf = cf_top(vp, false);
      if (!cf)
         return true;
      if (cf->has_else || !fold_jump_in_if(vp, cf))
         emit_label(vp, cf->has_else ? cf->label_b : cf->label_a);
      vp->cf_depth--;
      return true;

   case TGSI_OPCODE_BGNLOOP:
      cf = cf_push(vp, true);
      if (cf)
         emit_label(vp, cf->label_a);
      return true;

   case TGSI_OPCODE_ENDLOOP:
      cf = cf_top(vp, true);
      if (!cf)
         return true;
      if (!rotate_loop(vp, cf))
         emit_branch(vp, cf->label_a, true, true, true, VP_SWZ_X);
      emit_label(vp, cf->label_b);
      vp->cf_depth--;
      return true;

   case TGSI_OPCODE_BRK:
      cf = cf_loop(vp);
      if (cf)
         emit_branch(vp, cf->label_b, true, true, true, VP_SWZ_X);
      return true;

   case TGSI_OPCODE_CONT:
      cf = cf_loop(vp);
      if (cf)
         emit_branch(vp, cf->label_a, true, true, true, VP_SWZ_X);
      return true;

   default:
      return false;
   }
}

/* Replace the labels by instruction indices. A branch to the end of the
 * program, and a program that ends with a branch, need one more instruction
 * to land on: NVIDIA's compiler ends such a program with an empty one (the
 * one with the end-of-program bit). */
static void
resolve_labels(struct grate_vp_shader *vp)
{
   unsigned *index =
      CALLOC(vp->num_labels ? vp->num_labels : 1, sizeof(*index));
   if (!index) {
      vp->failed = true;
      return;
   }

   unsigned count = 0;
   list_for_each_entry_safe (struct vp_instr, instr, &vp->instructions,
                             link) {
      if (instr->is_label) {
         index[instr->label] = count;
         list_del(&instr->link);
         FREE(instr);
      } else {
         count++;
      }
   }

   bool landing = false;
   for (unsigned i = 0; i < vp->num_labels; i++)
      landing |= index[i] == count;
   if (count > 0 &&
       list_last_entry(&vp->instructions, struct vp_instr, link)->scalar.op ==
          VP_SCALAR_OP_BRA)
      landing = true;
   if (landing) {
      struct vp_instr *nop = emit_packed(emit_vNOP(), emit_sNOP());
      list_addtail(&nop->link, &vp->instructions);
   }

   list_for_each_entry (struct vp_instr, instr, &vp->instructions, link)
      if (instr->scalar.op == VP_SCALAR_OP_BRA)
         instr->branch.target = index[instr->branch.label];

   FREE(index);
}

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi)
{
   list_inithead(&vp->instructions);
   vp->output_mask = 0;
   vp->num_immediates = 0;
   vp->num_temps = 0;
   vp->failed = false;
   vp->cf_depth = 0;
   vp->num_labels = 0;

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
            if (tgsi_control_flow(vp, &tgsi->FullToken.FullInstruction))
               break;
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

   if (vp->cf_depth) {
      fprintf(stderr, "GRATE VERTEX: unbalanced control flow\n");
      vp->failed = true;
   }
   if (!vp->failed)
      resolve_labels(vp);
}
