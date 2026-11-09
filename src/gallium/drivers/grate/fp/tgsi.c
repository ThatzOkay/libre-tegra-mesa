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

/* Save one scalar through R0. Surface-4 ST_R20 preserves its FP20 value;
 * an FX10 source is converted before the store. */
static void
emit_spill_scalar(struct grate_fp_shader *fp, int loc, uint16_t args)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   struct fp_alu_src_operand src = {
      .index = loc & 0x7f,
      .datatype = (loc & TEMP_FX10) ? FP_DATATYPE_FIXED10 : FP_DATATYPE_FP20,
      .sub_reg_select_high = !!(loc & TEMP_HIGH),
   };
   struct fp_alu_dst_operand dst = {
      .index = 0,
      .write_low_sub_reg = true,
      .write_high_sub_reg = true,
   };

   list_inithead(&inst->link);
   list_inithead(&alu->link);
   alu->slots[0] = fp_alu_sMOV(dst, src);
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   inst->alu_sched.num_instructions = 1;
   inst->dw.enable = true;
   inst->dw.spill = true;
   inst->dw.index = GRATE_FP_SPILL_SURFACE;
   inst->dw.spill_args = args;
   inst->dw.write_killed = true;
   inst->dw.cache_persistent = true;
   list_addtail(&alu->link, &fp->alu_instructions);
   list_addtail(&inst->link, &fp->fp_instructions);
}

/* The load on row 3 supplies R12 to this round's ALU. Restore only the
 * original half of an FX10 destination, preserving its other scalar. */
static void
emit_reload_scalar(struct grate_fp_shader *fp, int loc, uint16_t args)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   struct fp_alu_dst_operand dst = {
      .index = loc & 0x7f,
      .write_low_sub_reg = !(loc & TEMP_HIGH),
      .write_high_sub_reg = !(loc & TEMP_FX10) || (loc & TEMP_HIGH),
   };

   list_inithead(&inst->link);
   list_inithead(&alu->link);
   alu->slots[0] = fp_alu_sMOV(dst, fp_alu_src_row(12));
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   inst->alu_sched.num_instructions = 1;
   inst->spill_load = true;
   /* OP=LD_R20, READ_KILLED, SURF=4; ARGS encode the load address. */
   inst->pseq = 0x01240000 | args;
   list_addtail(&alu->link, &fp->alu_instructions);
   list_addtail(&inst->link, &fp->fp_instructions);
}

static unsigned
scalar_mask(int loc)
{
   return (loc & TEMP_FX10) ? ((loc & TEMP_HIGH) ? 2 : 1) : 3;
}

/* The old value is dead if a write replaces it before its next read. Such
 * values, and values never read again, are the cheapest spill victims. */
static unsigned
next_scalar_use(const struct grate_fp_shader *fp, unsigned temp,
                unsigned comp)
{
   const struct tgsi_full_instruction *first = fp->current_instruction + 1;
   for (const struct tgsi_full_instruction *inst = first; inst < fp->code_end;
        inst++) {
      for (unsigned s = 0; s < inst->Instruction.NumSrcRegs; s++) {
         const struct tgsi_src_register *src = &inst->Src[s].Register;
         if (src->File == TGSI_FILE_TEMPORARY && src->Index == temp &&
             (tgsi_util_get_inst_usage_mask(inst, s) & (1 << comp)))
            return inst - first;
      }
      for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
         const struct tgsi_dst_register *dst = &inst->Dst[d].Register;
         if (dst->File == TGSI_FILE_TEMPORARY && dst->Index == temp &&
             (dst->WriteMask & (1 << comp)))
            return fp->code_end - first;
      }
   }
   return fp->code_end - first;
}

/* Sources of this TGSI instruction and destinations already allocated for
 * it cannot be evicted: their packets have not executed yet. */
static bool
spill_victim(struct grate_fp_shader *fp)
{
   if (!fp->current_instruction || !fp->code_end)
      return false;
   unsigned free_slot;
   for (free_slot = 0; free_slot < GRATE_FP_MAX_SPILLS; free_slot++)
      if (!fp->spill_used[free_slot])
         break;

   bool found = false;
   unsigned victim_temp = 0, victim_comp = 0, distance = 0;
   for (unsigned t = 0; t < fp->num_temporaries; t++) {
      for (unsigned c = 0; c < 4; c++) {
         int loc = fp->temp_src[t][c];
         unsigned reg = loc & 0x7f;
         unsigned mask = scalar_mask(loc);

         if (loc < 0 || reg < 4 || reg > 15 || reg == 12 || fp->pinned[reg] ||
             fp->claimed[reg])
            continue;
         if (fp->spill_slot[t][c] < 0 && free_slot == GRATE_FP_MAX_SPILLS)
            continue;
         bool aliased = false;
         for (unsigned u = 0; u < fp->num_temporaries; u++) {
            for (unsigned k = 0; k < 4; k++) {
               int other = fp->temp_src[u][k];
               if ((u != t || k != c) && other >= 0 &&
                   (other & 0x7f) == reg && (scalar_mask(other) & mask))
                  aliased = true;
            }
         }
         if (aliased)
            continue;
         unsigned next = next_scalar_use(fp, t, c);
         if (!found || next > distance) {
            found = true;
            distance = next;
            victim_temp = t;
            victim_comp = c;
         }
      }
   }
   if (!found)
      return false;
   int loc = fp->temp_src[victim_temp][victim_comp];
   unsigned reg = loc & 0x7f;
   unsigned mask = scalar_mask(loc);
   int cached = fp->spill_slot[victim_temp][victim_comp];
   unsigned slot = cached >= 0 ? (unsigned)cached : free_slot;
   if (cached < 0) {
      emit_spill_scalar(fp, loc, 12 | (slot << 4));
      fp->spill_used[slot] = true;
      fp->spill_fx10[slot] = !!(loc & TEMP_FX10);
      fp->spill_slot[victim_temp][victim_comp] = slot;
      fp->info.num_spill_slots = MAX2(fp->info.num_spill_slots, slot + 1);
   }
   fp->temp_src[victim_temp][victim_comp] = -2 - (int)slot;
   if (fp->temp_dst[victim_temp][victim_comp] == loc)
      fp->temp_dst[victim_temp][victim_comp] = -2 - (int)slot;
   fp->occupied[reg] &= ~mask;
   fp->reusable[reg] &= ~mask;
   return true;
}

/* Take a register of `regs` for a scalar: a whole register (FP20), or one
 * half of it (FX10). Prefer a register whose value this instruction reads
 * for the last time. Returns the encoded location, or -1. */
static int
allocate_in_pool(struct grate_fp_shader *fp, const unsigned *regs,
                 unsigned num_regs, bool fx10)
{
   for (unsigned pass = 0; pass < 2; pass++) {
      for (unsigned n = 0; n < num_regs; n++) {
         unsigned reg = regs[n];

         for (unsigned half = 0; half < (fx10 ? 2u : 1u); half++) {
            unsigned mask = fx10 ? 1u << half : 3u;
            bool free_slot = pass == 0 ? (fp->reusable[reg] & mask) == mask &&
                                            !(fp->claimed[reg] & mask)
                                       : !(fp->occupied[reg] & mask);

            if (!free_slot)
               continue;

            fp->occupied[reg] |= mask;
            fp->claimed[reg] |= mask;
            return fx10 ? (reg | TEMP_FX10 | (half ? TEMP_HIGH : 0)) : reg;
         }
      }
   }
   return -1;
}

/* Comparison results have their own boolean bank. Sampled UNORM colours
 * retain their exact FX10 representation through MOV/CMP, while arithmetic
 * uses FP20. R0-R3 become available after the last interpolation/TEX fetch.
 * R4-R15 (three additional register rows) hold a value for as long as it
 * is wanted, whatever the fetches and interpolations do, like in the
 * programs of NVIDIA's compiler: the program tells the hardware how many
 * rows it uses (GLOBAL_PIX_ATTR.NUM_ROWS, see grate_compile_fs()). */
static int
allocate_scalar(struct grate_fp_shader *fp, unsigned kind)
{
   static const unsigned overflow[] = {4,  5,  6,  7,  8,  9,
                                       10, 11, 12, 13, 14, 15};
   static const unsigned spill_regs[] = {4,  5,  6,  7,  8, 9,
                                         10, 11, 13, 14, 15};
   static const unsigned rows[] = {0, 1, 2, 3};
   static const unsigned spill_rows[] = {1, 2, 3};
   static const unsigned banks[] = {64, 65, 66, 67, 68, 69, 70, 71};
   int loc;

   if (kind == 2) {
      /* one comparison result per bank register */
      for (unsigned n = 0; n < ARRAY_SIZE(banks); n++) {
         if (!fp->occupied[banks[n]]) {
            fp->occupied[banks[n]] = fp->claimed[banks[n]] = 1;
            return banks[n] | TEMP_FX10;
         }
      }
   } else {
      bool fx10 = kind == 1;

      if (fp->late_rows && !fp->avoid_rows) {
         loc = fp->spilling
                  ? allocate_in_pool(fp, spill_rows, ARRAY_SIZE(spill_rows),
                                     fx10)
                  : allocate_in_pool(fp, rows, ARRAY_SIZE(rows), fx10);
         if (loc >= 0)
            return loc;
      }
      const unsigned *pool = fp->spilling ? spill_regs : overflow;
      unsigned num_regs =
         fp->spilling ? ARRAY_SIZE(spill_regs) : ARRAY_SIZE(overflow);
      loc = allocate_in_pool(fp, pool, num_regs, fx10);
      if (loc >= 0)
         return loc;
      while (fp->spilling && spill_victim(fp)) {
         loc = allocate_in_pool(fp, pool, num_regs, fx10);
         if (loc >= 0)
            return loc;
      }
   }

   if (!fp->allocation_failed && fp->spilling)
      fprintf(stderr, "GRATE FRAG: scalar register bank exhausted\n");
   fp->allocation_failed = true;
   fp->register_pressure = true;
   return 31;
}

/* A fetch is a TEX and also the interpolation of a varying. A value that is
 * alive across either lives in a register that neither overwrites: R4-R15
 * (see allocate_scalar()), or, to keep the program in fewer register rows,
 * one of R0-R3 that nothing overwrites while the value is wanted. Rows
 * are overwritten by every interpolation (R<k> for each
 * interpolated channel k) and, once the instruction is done reading its
 * operands, by the texture fetch (R2/R3, plus R0/R1 when the coordinate is
 * staged); `clobber` applies to every value alive when the instruction
 * starts, `clobber_after` only to those that outlive it. Pick a row (a whole
 * row for an FP20 value, one half of it for an FX10 value) for the value
 * (temp, comp) born at instruction n that nothing overwrites while the value
 * is still wanted. Returns -1 when there is none.
 */
static int
allocate_coordinate_row(struct grate_fp_shader *fp, const uint8_t *clobber,
                        const uint8_t *clobber_after,
                        uint8_t (*live_after)[GRATE_FP_MAX_TEMPS],
                        unsigned count, unsigned n, unsigned temp,
                        unsigned comp, bool fx10, bool is_tex)
{
   /* A fetch result is already in R2/R3, channel c in half c % 2 of row
    * 2 + c / 2. Leave it there if nothing clobbers that row while the value
    * is wanted, otherwise move it to R0/R1, which the fetch has finished
    * reading. */
   for (unsigned attempt = 0; attempt < (is_tex ? 2u : 1u); attempt++) {
      for (unsigned reg = 0; reg < 4; reg++) {
         if (fp->spilling && reg == 0)
            continue;
         if (is_tex && attempt == 0 ? reg != 2 + comp / 2
                                    : (is_tex && reg >= 2))
            continue;
         if ((clobber[n] & (1 << reg)) && !(is_tex && reg < 2))
            continue;

         for (unsigned half = 0; half < (fx10 ? 2u : 1u); half++) {
            unsigned mask = fx10 ? 1u << half : 3u;

            if (is_tex && attempt == 0 && half != comp % 2)
               continue;
            /* A row whose value this instruction reads for the last time can
             * take the result, like allocate_scalar() does. */
            bool free_slot = !(fp->occupied[reg] & mask) ||
                             ((fp->reusable[reg] & mask) == mask &&
                              !(fp->claimed[reg] & mask));
            bool safe = true;

            if (!free_slot)
               continue;

            /* live_after[j] holds what is live when instruction j starts, so
             * j + 1 is what outlives it. */
            for (unsigned j = n + 1;
                 j < count && (live_after[j][temp] & (1 << comp)); j++) {
               safe &= !(clobber[j] & (1 << reg));
               if (live_after[j + 1][temp] & (1 << comp))
                  safe &= !(clobber_after[j] & (1 << reg));
            }
            if (!safe)
               continue;

            fp->occupied[reg] |= mask;
            fp->claimed[reg] |= mask;
            return fx10 ? (reg | TEMP_FX10 | (half ? TEMP_HIGH : 0)) : reg;
         }
      }
   }
   return -1;
}

/* The register for the value (temp, comp) of kind `kind` born at
 * instruction n: if the value outlives a fetch or an interpolation, a row
 * that survives it when there is one, else any register of the pool. */
static int
across_fetch(struct grate_fp_shader *fp, const uint8_t *clobber,
             const uint8_t *clobber_after,
             uint8_t (*live_after)[GRATE_FP_MAX_TEMPS], unsigned count,
             unsigned n, unsigned temp, unsigned comp, unsigned kind,
             bool is_tex, bool crosses)
{
   if (crosses && kind != 2 && !fp->avoid_rows) {
      int loc = allocate_coordinate_row(fp, clobber, clobber_after,
                                        live_after, count, n, temp, comp,
                                        kind == 1, is_tex);
      if (loc >= 0)
         return loc;
   }
   return allocate_scalar(fp, kind);
}

static struct fp_alu_dst_operand
fp_alu_dst(struct grate_fp_shader *fp, const struct tgsi_dst_register *dst,
           int subreg, bool saturate)
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
      assert(dst->File == TGSI_FILE_TEMPORARY &&
             dst->Index <= GRATE_FP_MAX_TEMPS);
      if (fp->temp_dst[dst->Index][subreg] < 0) {
         /* Internal interpolation staging lives only within this instruction.
          */
         assert(dst->Index == fp->num_temporaries);
         fp->temp_dst[dst->Index][subreg] = allocate_scalar(fp, false);
         fp->temp_src[dst->Index][subreg] = fp->temp_dst[dst->Index][subreg];
      }
      int location = fp->temp_dst[dst->Index][subreg];
      ret.index = location & 0x7f;
      ret.write_low_sub_reg = !(location & TEMP_HIGH);
      ret.write_high_sub_reg =
         !(location & TEMP_FX10) || (location & TEMP_HIGH);
   }

   ret.saturate = saturate;

   return ret;
}

/* With derivatives, r32, r40, r48 and r56 are the quad exchange registers
 * (NVIDIA's compiler allocates constants around them), so the constant slots
 * 0, 8, 16 and 24 are left out. */
static bool
fp_constant_reserved(const struct grate_fp_shader *fp, unsigned i)
{
   return fp->uses_derivatives && i % 8 == 0;
}

static struct fp_alu_src_operand
fp_constant(struct grate_fp_shader *fp, int word, float immediate)
{
   unsigned i;
   for (i = 0; i < fp->info.num_constants; i++) {
      if (!fp_constant_reserved(fp, i) &&
          fp->info.constants[i].word == word &&
          (word >= 0 || fp->info.constants[i].immediate == immediate))
         break;
   }
   if (i == fp->info.num_constants) {
      /* a new constant: skip the reserved slots */
      while (i < ARRAY_SIZE(fp->info.constants) &&
             fp_constant_reserved(fp, i)) {
         fp->info.constants[i].word = -1;
         fp->info.constants[i].immediate = 0.0f;
         i++;
      }
   }
   if (i >= ARRAY_SIZE(fp->info.constants)) {
      if (!fp->allocation_failed)
         fprintf(stderr, "GRATE FRAG: scalar constant bank exhausted\n");
      fp->allocation_failed = true;
      /* Spilling temporaries cannot make more constant registers. */
      fp->register_pressure = false;
      return fp_alu_src_zero();
   }
   if (i >= fp->info.num_constants) {
      fp->info.constants[i].word = word;
      fp->info.constants[i].immediate = immediate;
      fp->info.num_constants = i + 1;
   }
   struct fp_alu_src_operand value = {.index = 32 + i};
   return value;
}

static struct fp_alu_src_operand
fp_tgsi_src(struct grate_fp_shader *fp, const struct tgsi_src_register *src,
            unsigned component, struct fp_mfu_instr *mfu)
{
   unsigned swizzle[] = {src->SwizzleX, src->SwizzleY, src->SwizzleZ,
                         src->SwizzleW};
   unsigned comp = swizzle[component];
   struct fp_alu_src_operand value;

   switch (src->File) {
   case TGSI_FILE_INPUT:
      /* Interpolate the selected source channel, not the destination channel.
       */
      assert(mfu);
      assert(mfu->var[comp].op == FP_VAR_OP_NOP ||
             mfu->var[comp].tram_row == src->Index);
      mfu->var[comp].op = FP_VAR_OP_FP20;
      mfu->var[comp].tram_row = src->Index;
      fp->info.max_tram_row = MAX2(fp->info.max_tram_row, src->Index + 1);
      value = fp_alu_src_row(comp);
      break;
   case TGSI_FILE_IMMEDIATE: {
      float imm = fp->immediates[src->Index][comp];
      value = imm == 1.0f   ? fp_alu_src_one()
              : imm == 0.0f ? fp_alu_src_zero()
                            : fp_constant(fp, -1, imm);
      break;
   }
   case TGSI_FILE_CONSTANT:
      if (src->Indirect) {
         if (!fp->allocation_failed)
            fprintf(stderr,
                    "GRATE FRAG: indirect uniform indexing unsupported\n");
         fp->allocation_failed = true;
         fp->register_pressure = false;
         return fp_alu_src_zero();
      }
      value = fp_constant(fp, src->Index * 4 + comp, 0.0f);
      break;
   case TGSI_FILE_TEMPORARY:
      assert(src->Index <= GRATE_FP_MAX_TEMPS);
      int reg = fp->temp_src[src->Index][comp];
      assert(reg >= 0 || fp->allocation_failed);
      value = fp_alu_src_row(0);
      value.index = reg >= 0 ? reg & 0x7f : 31;
      value.datatype =
         (reg & TEMP_FX10) ? FP_DATATYPE_FIXED10 : FP_DATATYPE_FP20;
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
      mfu->mul[k].src[1] = k ? FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1
                             : FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;
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
   if (src->File == TGSI_FILE_INPUT ||
       (rhs && rhs->File == TGSI_FILE_INPUT) ||
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
         if (!operand || operand->File != TGSI_FILE_TEMPORARY)
            continue;
         unsigned swizzle[] = {operand->SwizzleX, operand->SwizzleY,
                               operand->SwizzleZ, operand->SwizzleW};
         for (unsigned c = 0; c < 4; c++) {
            int loc = fp->temp_src[operand->Index][swizzle[c]];
            if ((dst->WriteMask & (1 << c)) && loc >= 0 && (loc & 0x7f) < 4)
               output_stage = true;
         }
      }
   }
   bool direct_pack =
      output_stage && opcode == TGSI_OPCODE_MOV &&
      dst->WriteMask == TGSI_WRITEMASK_XYZW &&
      src->File == TGSI_FILE_TEMPORARY && src->SwizzleX == TGSI_SWIZZLE_X &&
      src->SwizzleY == TGSI_SWIZZLE_Y && src->SwizzleZ == TGSI_SWIZZLE_Z &&
      src->SwizzleW == TGSI_SWIZZLE_W;
   for (unsigned c = 0; direct_pack && c < 4; c++)
      direct_pack = fp->temp_src[src->Index][c] == (int)c;
   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   int alu_instrs = 0;
   list_inithead(&alu->link);
   for (int slot = 0; slot < 4; ++slot) {
      static const unsigned pack_order[] = {2, 3, 0, 1};
      int i = direct_pack ? pack_order[slot] : slot;
      if ((dst->WriteMask & (1 << i)) == 0)
         continue;

      struct fp_alu_src_operand a = fp_tgsi_src(fp, src, i, mfu);
      struct fp_alu_src_operand b =
         rhs ? fp_tgsi_src(fp, rhs, i, mfu) : fp_alu_src_one();
      struct fp_alu_instr op =
         fp_alu_sMOV(fp_alu_dst(fp, dst, i, saturate), a);
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
         op.condition = opcode == TGSI_OPCODE_SGE ? FP_CONDITION_GEQUAL
                                                  : FP_CONDITION_EQUAL;
         break;
      case TGSI_OPCODE_CMP:
         op.op = FP_ALU_OP_CSEL;
         op.src[2] = fp_tgsi_src(fp, third, i, mfu);
         break;
      case TGSI_OPCODE_MAD:
         op.src[2] = fp_tgsi_src(fp, third, i, mfu);
         break;
      case TGSI_OPCODE_MIN:
         /* like ADD, the second value is the third source: with it in the
          * second one the op returned min(a * b, 0) (measured) */
         op.op = FP_ALU_OP_MIN;
         op.src[1] = fp_alu_src_one();
         op.src[2] = b;
         break;
      default:
         break;
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
   list_addtail(&inst->link, &fp->fp_instructions);
}

/* TEX supports a vec2 coordinate and one base-level 2D sample. The sampler
 * reads its coordinate from R0/R1. A plain varying is interpolated straight
 * into them; a coordinate computed by earlier instructions (for example the
 * 0.5 * v + 0.5 remap in Tabula's wallpaper shader) is copied there first.
 * TEX returns packed fx10 channels in row registers 2/3.
 */
static void
emit_TEX(struct grate_fp_shader *fp, const struct tgsi_full_instruction *tgsi)
{
   const struct tgsi_src_register *coord = &tgsi->Src[0].Register;
   const struct tgsi_src_register *sampler = &tgsi->Src[1].Register;
   const struct tgsi_dst_register *dst = &tgsi->Dst[0].Register;
   assert(tgsi->Texture.Texture == TGSI_TEXTURE_2D);
   assert((coord->File == TGSI_FILE_INPUT && !coord->Negate &&
           !coord->Absolute) ||
          coord->File == TGSI_FILE_TEMPORARY ||
          coord->File == TGSI_FILE_IMMEDIATE);
   assert(coord->File != TGSI_FILE_INPUT ||
          (coord->SwizzleX == 0 && coord->SwizzleY == 1));
   assert(sampler->File == TGSI_FILE_SAMPLER && sampler->Index < 16);

   struct fp_instr *interp = CALLOC_STRUCT(fp_instr);
   list_inithead(&interp->link);
   if (coord->File != TGSI_FILE_INPUT) {
      struct fp_alu_instr_packet *stage = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&stage->link);
      for (unsigned i = 0; i < 2; i++) {
         struct fp_alu_dst_operand row = {
            .index = i,
            .write_low_sub_reg = true,
            .write_high_sub_reg = true,
         };
         stage->slots[i] = fp_alu_sMOV(row, fp_tgsi_src(fp, coord, i, NULL));
      }
      interp->alu_sched.address = list_length(&fp->alu_instructions);
      interp->alu_sched.num_instructions = 1;
      list_addtail(&stage->link, &fp->alu_instructions);
      list_addtail(&interp->link, &fp->fp_instructions);
   } else {
      struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
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
   }

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
      if (!(dst->WriteMask & (1 << i)))
         continue;
      unsigned channel = i;
      struct fp_alu_src_operand src = {
         .index = 2 + channel / 2,
         .datatype = FP_DATATYPE_FIXED10,
         .sub_reg_select_high = channel % 2,
      };
      alu->slots[slot++] =
         fp_alu_sMOV(fp_alu_dst(fp, dst, i, tgsi->Instruction.Saturate), src);
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

/* Copy component comp of src into the ALU register index, interpolating a
 * varying first. This also converts FX10 to FP20. absolute copies the
 * magnitude. */
static void
emit_load_scalar(struct grate_fp_shader *fp,
                 const struct tgsi_src_register *src, unsigned comp,
                 int index, bool absolute)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);
   struct fp_alu_instr_packet *copy = CALLOC_STRUCT(fp_alu_instr_packet);
   list_inithead(&copy->link);
   struct fp_alu_dst_operand g = {.index = index,
                                  .write_low_sub_reg = true,
                                  .write_high_sub_reg = true};
   struct tgsi_src_register one_comp = *src;
   one_comp.SwizzleX = one_comp.SwizzleY = one_comp.SwizzleZ =
      one_comp.SwizzleW = comp;
   struct fp_mfu_instr *interp = NULL;
   if (src->File == TGSI_FILE_INPUT) {
      /* a varying is interpolated first, into the row of its channel */
      interp = CALLOC_STRUCT(fp_mfu_instr);
      list_inithead(&interp->link);
      setup_interpolation(interp);
   }
   struct fp_alu_src_operand value = fp_tgsi_src(fp, &one_comp, 0, interp);
   value.absolute_value |= absolute;
   copy->slots[0] = fp_alu_sMOV(g, value);
   if (interp) {
      inst->mfu_sched.num_instructions = 1;
      inst->mfu_sched.address = list_length(&fp->mfu_instructions);
      list_addtail(&interp->link, &fp->mfu_instructions);
   }
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   inst->alu_sched.num_instructions = 1;
   list_addtail(&copy->link, &fp->alu_instructions);
   list_addtail(&inst->link, &fp->fp_instructions);
}

/* dFdx and dFdy over the pixel quad. Every pixel writes the value to the
 * exchange port r32; two packets later (the layout adds the empty one) the
 * four values of its 2x2 quad can be read back, whichever pixel wrote them:
 * r32 = top left, r40 = top right, r48 = bottom left, r56 = bottom right
 * (measured with ramps of tc.x and tc.y, see docs/data/gpu/derivatives).
 * The derivatives are coarse, one per quad: dFdx = r40 - r32 and, along the
 * hardware's raster y (downwards), dFdy = r48 - r32. The state tracker
 * multiplies the argument of DDY by the framebuffer's y-flip factor to get
 * the GL direction. The slots are read through the A port in one packet
 * (the C port returns the slot of a neighbouring quad). These registers
 * share their index space with the constants, so fp_constant() leaves r32,
 * r40, r48 and r56 out. */
static void
emit_derivative(struct grate_fp_shader *fp,
                const struct tgsi_full_instruction *tgsi, bool vertical)
{
   const struct tgsi_dst_register *dst = &tgsi->Dst[0].Register;
   const struct tgsi_src_register *src = &tgsi->Src[0].Register;

   for (unsigned c = 0; c < 4; c++) {
      if (!(dst->WriteMask & (1 << c)))
         continue;

      struct fp_instr *write = CALLOC_STRUCT(fp_instr);
      list_inithead(&write->link);
      struct fp_mfu_instr *mfu = NULL;
      if (src->File == TGSI_FILE_INPUT) {
         mfu = CALLOC_STRUCT(fp_mfu_instr);
         list_inithead(&mfu->link);
         setup_interpolation(mfu);
      }
      struct fp_alu_instr_packet *store = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&store->link);
      struct fp_alu_dst_operand port = {.index = 32,
                                        .write_low_sub_reg = true,
                                        .write_high_sub_reg = true};
      store->slots[0] = fp_alu_sMOV(port, fp_tgsi_src(fp, src, c, mfu));
      write->alu_sched.address = list_length(&fp->alu_instructions);
      write->alu_sched.num_instructions = 1;
      if (mfu) {
         write->mfu_sched.num_instructions = 1;
         write->mfu_sched.address = list_length(&fp->mfu_instructions);
         list_addtail(&mfu->link, &fp->mfu_instructions);
      }
      list_addtail(&store->link, &fp->alu_instructions);
      list_addtail(&write->link, &fp->fp_instructions);

      /* Read the two slots through the A port into scratch scalars. The C
       * port would return the slot of a neighbouring quad (measured). */
      assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
      struct tgsi_dst_register scratch = {0};
      scratch.File = TGSI_FILE_TEMPORARY;
      scratch.Index = fp->num_temporaries;
      scratch.WriteMask = TGSI_WRITEMASK_XY;
      struct fp_instr *read = CALLOC_STRUCT(fp_instr);
      list_inithead(&read->link);
      struct fp_alu_instr_packet *slots = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&slots->link);
      struct fp_alu_src_operand slot_of[2] = {
         {.index = vertical ? 48 : 40}, {.index = vertical ? 32 : 32}};
      for (unsigned k = 0; k < 2; k++)
         slots->slots[k] =
            fp_alu_sMOV(fp_alu_dst(fp, &scratch, k, false), slot_of[k]);
      read->alu_sched.address = list_length(&fp->alu_instructions);
      read->alu_sched.num_instructions = 1;
      list_addtail(&slots->link, &fp->alu_instructions);
      list_addtail(&read->link, &fp->fp_instructions);

      struct tgsi_src_register pair = {0};
      pair.File = TGSI_FILE_TEMPORARY;
      pair.Index = fp->num_temporaries;
      pair.SwizzleX = TGSI_SWIZZLE_X;
      pair.SwizzleY = TGSI_SWIZZLE_Y;
      struct fp_instr *calc = CALLOC_STRUCT(fp_instr);
      list_inithead(&calc->link);
      struct fp_alu_instr_packet *diff = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&diff->link);
      struct fp_alu_src_operand first = fp_tgsi_src(fp, &pair, 0, NULL);
      struct fp_alu_src_operand second = fp_tgsi_src(fp, &pair, 1, NULL);
      second.negate = true;
      struct fp_alu_instr op = fp_alu_sMOV(
         fp_alu_dst(fp, dst, c, tgsi->Instruction.Saturate), first);
      op.src[2] = second;
      diff->slots[0] = op;
      calc->alu_sched.address = list_length(&fp->alu_instructions);
      calc->alu_sched.num_instructions = 1;
      list_addtail(&diff->link, &fp->alu_instructions);
      list_addtail(&calc->link, &fp->fp_instructions);
      /* the scratch scalars are free for the next component */
      memset(fp->temp_src[fp->num_temporaries], -1,
             sizeof(fp->temp_src[0]));
      memset(fp->temp_dst[fp->num_temporaries], -1,
             sizeof(fp->temp_dst[0]));
   }
}

/* One SFU step. The SFU reads the row register `reg` (R0-R3 only) and
 * writes its result back into it, before the ALU of the same step runs, so
 * `alu` (if any) sees the result in `reg`. The multipliers are not used:
 * all-zero multiplier fields are what NVIDIA's programs have. */
static void
emit_sfu_step(struct grate_fp_shader *fp, enum fp_sfu_op op, unsigned reg,
              struct fp_alu_instr_packet *alu)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);
   struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
   list_inithead(&mfu->link);
   mfu->sfu.op = op;
   mfu->sfu.reg = reg;
   inst->mfu_sched.num_instructions = 1;
   inst->mfu_sched.address = list_length(&fp->mfu_instructions);
   list_addtail(&mfu->link, &fp->mfu_instructions);
   if (alu) {
      inst->alu_sched.num_instructions = 1;
      inst->alu_sched.address = list_length(&fp->alu_instructions);
      list_addtail(&alu->link, &fp->alu_instructions);
   }
   list_addtail(&inst->link, &fp->fp_instructions);
}

/* RCP, RSQ, SQRT, LG2, EX2 and POW of computed scalars, the way NVIDIA's
 * compiler does them (decoded from its output for division, sqrt, rsqrt,
 * exp2, log2 and pow, see docs/grate-register-bug-handoff.md): the SFU op
 * runs on a row register R0-R3 and replaces its value in place, one op per
 * step, and the ALU of the same step reads the result from that register.
 * So: copy the operand into a free row (this also converts FX10 to FP20),
 * run the op, and store the register into the destination in the same
 * step. EX2 is PREEX2 in one step, which builds 2^floor(v) * (1 + frac(v)),
 * and EX2 in the next, which corrects the mantissa to 2^frac(v). POW is
 * LG2, an ALU multiply by the exponent in the same step, PREEX2, EX2. The
 * SFU reads R0-R3 only (measured: R4 to R15 as its operand return nothing
 * useful), so the allocator keeps values that live across an SFU op out of
 * the rows (fp->avoid_rows). */
static void
emit_SFU(struct grate_fp_shader *fp, const struct tgsi_full_instruction *tgsi)
{
   unsigned opcode = tgsi->Instruction.Opcode;
   bool is_pow = opcode == TGSI_OPCODE_POW;
   const struct tgsi_dst_register *dst = &tgsi->Dst[0].Register;
   const struct tgsi_src_register *src = &tgsi->Src[0].Register;
   const struct tgsi_src_register *src_y =
      is_pow ? &tgsi->Src[1].Register : NULL;
   unsigned comp = src->SwizzleX;

   /* Free rows: not holding a value that is still wanted (the destination
    * has been claimed already, so it is not among them). */
   int free_rows[4];
   unsigned num_free = 0;
   for (unsigned reg = 0; reg < 4; reg++) {
      if (!fp->occupied[reg])
         free_rows[num_free++] = reg;
   }

   /* POW: the row that holds the exponent for the multiplication */
   int in_reg = num_free ? free_rows[0] : -1;
   int y_row = -1;
   if (is_pow && num_free > 1) {
      y_row = free_rows[1];
      if (src_y->File == TGSI_FILE_INPUT && in_reg == (int)src_y->SwizzleX) {
         /* the exponent's interpolation would overwrite the base */
         in_reg = free_rows[1];
         y_row = free_rows[0];
      }
   }
   if (in_reg < 0 || (is_pow && y_row < 0)) {
      if (!fp->allocation_failed)
         fprintf(stderr, "GRATE FRAG: no free register for the SFU op "
                         "for the SFU operation\n");
      fp->allocation_failed = true;
      fp->register_pressure = true;
      return;
   }

   /* the operands go into rows */
   emit_load_scalar(fp, src, comp, in_reg, opcode == TGSI_OPCODE_RSQ);
   if (is_pow)
      emit_load_scalar(fp, src_y, src_y->SwizzleX, y_row, false);

   /* the store of the result into the destination, in the last step */
   struct fp_alu_instr_packet *store = CALLOC_STRUCT(fp_alu_instr_packet);
   list_inithead(&store->link);
   unsigned slot = 0;
   for (unsigned i = 0; i < 4; i++) {
      if (!(dst->WriteMask & (1 << i)))
         continue;
      struct fp_alu_src_operand r = fp_alu_src_row(in_reg);
      r.datatype = FP_DATATYPE_FP20;
      store->slots[slot++] =
         fp_alu_sMOV(fp_alu_dst(fp, dst, i, tgsi->Instruction.Saturate), r);
   }

   switch (opcode) {
   case TGSI_OPCODE_RCP:
      emit_sfu_step(fp, FP_SFU_OP_RCP, in_reg, store);
      break;
   case TGSI_OPCODE_RSQ:
      emit_sfu_step(fp, FP_SFU_OP_RSQ, in_reg, store);
      break;
   case TGSI_OPCODE_SQRT:
      emit_sfu_step(fp, FP_SFU_OP_SQRT, in_reg, store);
      break;
   case TGSI_OPCODE_LG2:
      emit_sfu_step(fp, FP_SFU_OP_LG2, in_reg, store);
      break;
   case TGSI_OPCODE_EX2:
      emit_sfu_step(fp, FP_SFU_OP_PREEX2, in_reg, NULL);
      emit_sfu_step(fp, FP_SFU_OP_EX2, in_reg, store);
      break;
   case TGSI_OPCODE_POW: {
      struct fp_alu_instr_packet *scale = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&scale->link);
      struct fp_alu_dst_operand r = {.index = in_reg,
                                     .write_low_sub_reg = true,
                                     .write_high_sub_reg = true};
      struct fp_alu_src_operand x = fp_alu_src_row(in_reg);
      x.datatype = FP_DATATYPE_FP20;
      scale->slots[0] = fp_alu_sMOV(r, x);
      scale->slots[0].src[1] = fp_alu_src_row(y_row);
      scale->slots[0].src[1].datatype = FP_DATATYPE_FP20;
      emit_sfu_step(fp, FP_SFU_OP_LG2, in_reg, scale);
      emit_sfu_step(fp, FP_SFU_OP_PREEX2, in_reg, NULL);
      emit_sfu_step(fp, FP_SFU_OP_EX2, in_reg, store);
      break;
   }
   default:
      UNREACHABLE("not an SFU opcode");
   }

   /* the output write goes with the step that stores the colour */
   if (dst->File == TGSI_FILE_OUTPUT) {
      struct fp_instr *last =
         list_last_entry(&fp->fp_instructions, struct fp_instr, link);
      last->dw.enable = true;
      last->dw.index = dst->Index + 1;
      last->dw.src_regs = FP_DW_REGS_R2_R3;
   }
}

/* The component of src that TGSI component k reads. */
static unsigned
src_component(const struct tgsi_src_register *src, unsigned k)
{
   unsigned swizzle[] = {src->SwizzleX, src->SwizzleY, src->SwizzleZ,
                         src->SwizzleW};
   return swizzle[k];
}

/* NVIDIA uses SFU FRC on an FP20 row operand. FLR subtracts that
 * fraction from the original value; converting to FX10 is not FRC. */
static void
emit_rounding(struct grate_fp_shader *fp,
              const struct tgsi_full_instruction *tgsi)
{
   const struct tgsi_dst_register *dst = &tgsi->Dst[0].Register;
   const struct tgsi_src_register *src = &tgsi->Src[0].Register;
   bool floor = tgsi->Instruction.Opcode == TGSI_OPCODE_FLR;

   for (unsigned c = 0; c < 4; c++) {
      if (!(dst->WriteMask & (1 << c)))
         continue;
      unsigned comp = src_component(src, c);
      int row = -1;
      for (unsigned r = 0; r < 4; r++) {
         /* An interpolated original must survive until subtraction. */
         if (!fp->occupied[r] &&
             !(floor && src->File == TGSI_FILE_INPUT && r == comp)) {
            row = r;
            break;
         }
      }
      if (row < 0) {
         fp->allocation_failed = true;
         fp->register_pressure = true;
         return;
      }

      struct fp_alu_src_operand original = {0};
      if (floor) {
         if (src->File == TGSI_FILE_INPUT) {
            original = fp_alu_src_row(comp);
            original.negate = src->Negate;
            original.absolute_value = src->Absolute;
         } else {
            original = fp_tgsi_src(fp, src, c, NULL);
         }
      }
      emit_load_scalar(fp, src, comp, row, false);
      struct fp_alu_instr_packet *store = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&store->link);
      struct fp_alu_src_operand fraction = fp_alu_src_row(row);
      store->slots[0] =
         fp_alu_sMOV(fp_alu_dst(fp, dst, c, tgsi->Instruction.Saturate),
                     floor ? original : fraction);
      if (floor) {
         fraction.negate = true;
         store->slots[0].src[2] = fraction;
      }
      emit_sfu_step(fp, FP_SFU_OP_FRC, row, store);
   }
}

/* DP2, DP3 and DP4: the sum of a.k * b.k as a multiply and fused
 * multiply-adds, which leave the running sum in channel W of the scratch
 * temporary. Every step reads component k of both operands in all channels,
 * so the result lands in each channel of the destination. An MFU packet
 * interpolates one vec4 and every interpolation overwrites the rows of the
 * channels it reads, so a varying operand is copied into the scratch's X, Y
 * and Z first (DP2 and DP3 only: DP4 has no channel left for the sum). */
static void
emit_DP(struct grate_fp_shader *fp, const struct tgsi_full_instruction *tgsi)
{
   unsigned n = tgsi->Instruction.Opcode == TGSI_OPCODE_DP2   ? 2
                : tgsi->Instruction.Opcode == TGSI_OPCODE_DP3 ? 3
                                                              : 4;
   bool saturate = tgsi->Instruction.Saturate != 0;
   const struct tgsi_dst_register *dst = &tgsi->Dst[0].Register;
   struct tgsi_src_register src[2] = {tgsi->Src[0].Register,
                                      tgsi->Src[1].Register};
   bool is_input[2] = {src[0].File == TGSI_FILE_INPUT,
                       src[1].File == TGSI_FILE_INPUT};
   bool same_input = is_input[0] && is_input[1] &&
                     !memcmp(&src[0], &src[1], sizeof(src[0]));
   if ((is_input[0] || is_input[1]) &&
       (n == 4 || (is_input[0] && is_input[1] && !same_input))) {
      if (!fp->allocation_failed)
         fprintf(stderr, "GRATE FRAG: DP4 of a varying, or DP of two "
                         "different varyings, is not supported\n");
      fp->allocation_failed = true;
      return;
   }

   assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
   /* the scratch must not take the register of an operand that a later step
    * still reads, even if this instruction reads it for the last time */
   memset(fp->reusable, 0, sizeof(fp->reusable));
   struct tgsi_dst_register stage = {0};
   stage.File = TGSI_FILE_TEMPORARY;
   stage.Index = fp->num_temporaries;
   stage.WriteMask = TGSI_WRITEMASK_XYZ;
   for (unsigned s = 0; s < 2; s++) {
      if (!is_input[s] || (s == 1 && same_input))
         continue;
      emit_vALU(fp, &stage, false, &src[s], NULL, NULL, TGSI_OPCODE_MOV);
      src[s].File = TGSI_FILE_TEMPORARY;
      src[s].Index = stage.Index;
      src[s].Negate = src[s].Absolute = 0;
      src[s].SwizzleX = TGSI_SWIZZLE_X;
      src[s].SwizzleY = TGSI_SWIZZLE_Y;
      src[s].SwizzleZ = TGSI_SWIZZLE_Z;
      src[s].SwizzleW = TGSI_SWIZZLE_W;
   }
   if (same_input)
      src[1] = src[0];

   struct tgsi_dst_register sum = stage;
   sum.WriteMask = TGSI_WRITEMASK_W;
   struct tgsi_src_register sum_src = {0};
   sum_src.File = TGSI_FILE_TEMPORARY;
   sum_src.Index = stage.Index;
   sum_src.SwizzleX = sum_src.SwizzleY = sum_src.SwizzleZ =
      sum_src.SwizzleW = TGSI_SWIZZLE_W;

   for (unsigned k = 0; k < n; k++) {
      struct tgsi_src_register a = src[0];
      struct tgsi_src_register b = src[1];
      a.SwizzleX = a.SwizzleY = a.SwizzleZ = a.SwizzleW =
         src_component(&src[0], k);
      b.SwizzleX = b.SwizzleY = b.SwizzleZ = b.SwizzleW =
         src_component(&src[1], k);
      if (k == n - 1)
         emit_vALU(fp, dst, saturate, &a, &b, &sum_src, TGSI_OPCODE_MAD);
      else if (k == 0)
         emit_vALU(fp, &sum, false, &a, &b, NULL, TGSI_OPCODE_MUL);
      else
         emit_vALU(fp, &sum, false, &a, &b, &sum_src, TGSI_OPCODE_MAD);
   }
}

static void
emit_tgsi_instr(struct grate_fp_shader *fp, const struct tgsi_full_instruction *inst)
{
   bool saturate = inst->Instruction.Saturate != 0;
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      if (inst->Src[i].Register.File == TGSI_FILE_CONSTANT &&
          inst->Src[i].Register.Dimension)
         assert(inst->Src[i].Dimension.Index == 0 &&
                !inst->Src[i].Dimension.Indirect);
   }

   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_TEX:
      emit_TEX(fp, inst);
      break;
   case TGSI_OPCODE_DDX:
   case TGSI_OPCODE_DDY:
      emit_derivative(fp, inst, inst->Instruction.Opcode == TGSI_OPCODE_DDY);
      break;
   case TGSI_OPCODE_FRC:
   case TGSI_OPCODE_FLR:
      emit_rounding(fp, inst);
      break;
   case TGSI_OPCODE_RCP:
   case TGSI_OPCODE_RSQ:
   case TGSI_OPCODE_SQRT:
   case TGSI_OPCODE_LG2:
   case TGSI_OPCODE_EX2:
   case TGSI_OPCODE_POW:
      emit_SFU(fp, inst);
      break;
   case TGSI_OPCODE_MOV:
      emit_vALU(fp, &inst->Dst[0].Register, saturate, &inst->Src[0].Register,
                NULL, NULL, TGSI_OPCODE_MOV);
      break;

   case TGSI_OPCODE_ADD:
   case TGSI_OPCODE_SLT:
   case TGSI_OPCODE_SGE:
   case TGSI_OPCODE_SEQ:
   case TGSI_OPCODE_MIN:
   case TGSI_OPCODE_MUL: {
      const struct tgsi_src_register *a = &inst->Src[0].Register;
      const struct tgsi_src_register *b = &inst->Src[1].Register;
      struct tgsi_src_register staged = {0};
      if (a->File == TGSI_FILE_INPUT && b->File == TGSI_FILE_INPUT &&
          a->Index != b->Index) {
         /* One MFU packet interpolates one vec4. Stage A in allocator-owned
          * scratch scalars before loading B. */
         assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
         struct tgsi_dst_register scratch = {0};
         scratch.File = TGSI_FILE_TEMPORARY;
         scratch.Index = fp->num_temporaries;
         scratch.WriteMask = inst->Dst[0].Register.WriteMask;
         emit_vALU(fp, &scratch, false, a, NULL, NULL, TGSI_OPCODE_MOV);
         staged.File = TGSI_FILE_TEMPORARY;
         staged.Index = fp->num_temporaries;
         staged.SwizzleX = TGSI_SWIZZLE_X;
         staged.SwizzleY = TGSI_SWIZZLE_Y;
         staged.SwizzleZ = TGSI_SWIZZLE_Z;
         staged.SwizzleW = TGSI_SWIZZLE_W;
         a = &staged;
      }
      emit_vALU(fp, &inst->Dst[0].Register, saturate, a, b, NULL,
                inst->Instruction.Opcode);

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
         if (!needs_stage)
            continue;
         assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS && !staged.File);
         struct tgsi_dst_register scratch = {0};
         scratch.File = TGSI_FILE_TEMPORARY;
         scratch.Index = fp->num_temporaries;
         scratch.WriteMask = inst->Dst[0].Register.WriteMask;
         emit_vALU(fp, &scratch, false, *inputs[i], NULL, NULL,
                   TGSI_OPCODE_MOV);
         staged.File = TGSI_FILE_TEMPORARY;
         staged.Index = scratch.Index;
         staged.SwizzleX = TGSI_SWIZZLE_X;
         staged.SwizzleY = TGSI_SWIZZLE_Y;
         staged.SwizzleZ = TGSI_SWIZZLE_Z;
         staged.SwizzleW = TGSI_SWIZZLE_W;
         *inputs[i] = &staged;
      }
      emit_vALU(fp, &inst->Dst[0].Register, saturate, a, b, c,
                TGSI_OPCODE_CMP);
      break;
   }

   case TGSI_OPCODE_DP2:
   case TGSI_OPCODE_DP3:
   case TGSI_OPCODE_DP4:
      emit_DP(fp, inst);
      break;

   case TGSI_OPCODE_LRP: {
      /* LRP(a, b, c) = a * b + (1 - a) * c, as two multiply-adds around a
       * scratch t = c - a * c. An MFU packet interpolates one vec4, so the
       * varyings among the operands have to be the same one. */
      const struct tgsi_src_register *a = &inst->Src[0].Register;
      const struct tgsi_src_register *b = &inst->Src[1].Register;
      const struct tgsi_src_register *c = &inst->Src[2].Register;
      const struct tgsi_src_register *operands[] = {a, b, c};
      int input = -1;
      for (unsigned i = 0; i < 3; i++) {
         if (operands[i]->File != TGSI_FILE_INPUT)
            continue;
         if (input >= 0 && input != operands[i]->Index) {
            if (!fp->allocation_failed)
               fprintf(stderr, "GRATE FRAG: LRP of two different varyings is "
                               "not supported\n");
            fp->allocation_failed = true;
            return;
         }
         input = operands[i]->Index;
      }
      assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
      /* t must not take the register of an operand that the second step
       * still reads, even if this instruction reads it for the last time. */
      memset(fp->reusable, 0, sizeof(fp->reusable));
      struct tgsi_dst_register scratch = {0};
      scratch.File = TGSI_FILE_TEMPORARY;
      scratch.Index = fp->num_temporaries;
      scratch.WriteMask = inst->Dst[0].Register.WriteMask;
      struct tgsi_src_register minus_a = *a;
      minus_a.Negate = !a->Negate;
      emit_vALU(fp, &scratch, false, &minus_a, c, c, TGSI_OPCODE_MAD);
      struct tgsi_src_register t = {0};
      t.File = TGSI_FILE_TEMPORARY;
      t.Index = scratch.Index;
      t.SwizzleX = TGSI_SWIZZLE_X;
      t.SwizzleY = TGSI_SWIZZLE_Y;
      t.SwizzleZ = TGSI_SWIZZLE_Z;
      t.SwizzleW = TGSI_SWIZZLE_W;
      emit_vALU(fp, &inst->Dst[0].Register, saturate, a, b, &t,
                TGSI_OPCODE_MAD);
      break;
   }

   default:
      /* An instruction this compiler cannot express is not a bug of the
       * application: the compiler is a straight-line translator for a small
       * part of TGSI (no control flow, no dot products beyond DP3/DP4, ...)
       * and real programs, Firefox's WebRender above all, use more than it
       * knows. Refuse the shader like a register shortage does: no shader
       * is created, the draws that need it are skipped (see grate_draw_vbo())
       * and the application lives on. Aborting here took the whole process,
       * and with it the browser's compositor, down for one missing opcode.
       * The message stays, naming the opcode to implement next; it is
       * printed even after an earlier failure, since a missing opcode says
       * more than a register shortage does. */
      fprintf(stderr, "GRATE FRAG TGSI UNIMPLEMENTED: 0x%02x\n",
              inst->Instruction.Opcode);
      fp->allocation_failed = true;
      break;
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

/* Stream a direct product of two varyings through R0/R1. The linker places
 * the selected A component in TRAM.x and B in TRAM.y, so each MFU fetch can
 * read both operands without a persistent global-register temporary. Keep
 * the accumulated packed output in R2/R3 until the final component.
 */
static bool
emit_direct_varying_product(struct grate_fp_shader *fp,
                            const struct tgsi_full_instruction *code,
                            unsigned count)
{
   if (count != 1 || code->Instruction.Opcode != TGSI_OPCODE_MUL ||
       code->Instruction.NumDstRegs != 1 ||
       code->Instruction.NumSrcRegs != 2 ||
       code->Dst[0].Register.File != TGSI_FILE_OUTPUT ||
       code->Dst[0].Register.WriteMask != TGSI_WRITEMASK_XYZW)
      return false;
   const struct tgsi_src_register *a = &code->Src[0].Register;
   const struct tgsi_src_register *b = &code->Src[1].Register;
   if (a->File != TGSI_FILE_INPUT || b->File != TGSI_FILE_INPUT ||
       a->Index == b->Index || a->Index >= 15 || b->Index >= 15 ||
       a->Indirect || b->Indirect)
      return false;
   unsigned swizzle[2][4] = {
      {a->SwizzleX, a->SwizzleY, a->SwizzleZ, a->SwizzleW},
      {b->SwizzleX, b->SwizzleY, b->SwizzleZ, b->SwizzleW},
   };
   fp->info.num_inputs = 8;
   fp->info.max_tram_row = 8;
   for (unsigned c = 0; c < 4; c++) {
      for (unsigned operand = 0; operand < 2; operand++) {
         unsigned row = c * 2 + operand;
         unsigned export = operand ? b->Index : a->Index;
         fp->info.inputs[row].src = LINK_SRC(export + 1) | (row << 9);
         fp->info.inputs[row].dst =
            LINK_DST(swizzle[operand][c], operand, LINK_DST_FP20);
      }
      struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
      struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
      struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&inst->link);
      list_inithead(&mfu->link);
      list_inithead(&alu->link);
      setup_interpolation(mfu);
      mfu->var[0].op = mfu->var[1].op = FP_VAR_OP_FP20;
      mfu->var[0].tram_row = c * 2;
      mfu->var[1].tram_row = c * 2 + 1;
      inst->mfu_sched.address = list_length(&fp->mfu_instructions);
      inst->mfu_sched.num_instructions = 1;
      inst->alu_sched.address = list_length(&fp->alu_instructions);
      inst->alu_sched.num_instructions = 1;
      struct fp_alu_src_operand va = fp_alu_src_row(0);
      struct fp_alu_src_operand vb = fp_alu_src_row(1);
      va.negate = a->Negate;
      va.absolute_value = a->Absolute;
      vb.negate = b->Negate;
      vb.absolute_value = b->Absolute;
      alu->slots[0] = fp_alu_sMOV(fp_alu_dst(fp, &code->Dst[0].Register, c,
                                             code->Instruction.Saturate),
                                  va);
      alu->slots[0].src[1] = vb;
      if (c == 3) {
         inst->dw.enable = 1;
         inst->dw.index = code->Dst[0].Register.Index + 1;
         inst->dw.src_regs = FP_DW_REGS_R2_R3;
      }
      list_addtail(&mfu->link, &fp->mfu_instructions);
      list_addtail(&alu->link, &fp->alu_instructions);
      list_addtail(&inst->link, &fp->fp_instructions);
   }
   return true;
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

/* The index of an immediate that has `value` in some channel, and that
 * channel; appends one if the shader has none. */
static unsigned
find_immediate(struct grate_fp_shader *fp, float value, unsigned *comp)
{
   for (unsigned i = 0; i < fp->num_immediates; i++) {
      for (unsigned c = 0; c < 4; c++) {
         if (fp->immediates[i][c] == value) {
            *comp = c;
            return i;
         }
      }
   }

   assert(fp->num_immediates < GRATE_FP_MAX_IMMEDIATES);
   unsigned index = fp->num_immediates++;
   for (unsigned c = 0; c < 4; c++)
      fp->immediates[index][c] = c == 0 ? value : 0.0f;
   *comp = 0;
   return index;
}

static void
set_src_swizzle(struct tgsi_src_register *src, unsigned x, unsigned y,
                unsigned z, unsigned w)
{
   src->SwizzleX = x;
   src->SwizzleY = y;
   src->SwizzleZ = z;
   src->SwizzleW = w;
}

static bool
reads_input(const struct tgsi_full_instruction *inst)
{
   for (unsigned s = 0; s < inst->Instruction.NumSrcRegs; s++) {
      if (inst->Src[s].Register.File == TGSI_FILE_INPUT)
         return true;
   }
   return false;
}

static bool
reads_temp(const struct tgsi_full_instruction *inst, unsigned index)
{
   for (unsigned s = 0; s < inst->Instruction.NumSrcRegs; s++)
      if (inst->Src[s].Register.File == TGSI_FILE_TEMPORARY &&
          inst->Src[s].Register.Index == index)
         return true;
   return false;
}

static bool
writes_temp(const struct tgsi_full_instruction *inst, unsigned index)
{
   for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++)
      if (inst->Dst[d].Register.File == TGSI_FILE_TEMPORARY &&
          inst->Dst[d].Register.Index == index)
         return true;
   return false;
}

/* The hardware has EQUAL, GREATER and GEQUAL conditions but no NOTEQUAL:
 * SNE(a, b) is SEQ(a, b) followed by 1 - x. */
static void
lower_sne(struct grate_fp_shader *fp, struct util_dynarray *instructions)
{
   unsigned count =
      util_dynarray_num_elements(instructions, struct tgsi_full_instruction);
   struct tgsi_full_instruction *code = instructions->data;
   /* The common case: the result only selects in CMP(-|x|, a, b). Not
    * equal then is equal with a and b swapped, and the comparison stays
    * in its own register bank, where it survives a texture fetch. */
   for (unsigned n = 0; n < count; n++) {
      if (code[n].Instruction.Opcode != TGSI_OPCODE_SNE ||
          code[n].Dst[0].Register.File != TGSI_FILE_TEMPORARY)
         continue;

      unsigned temp = code[n].Dst[0].Register.Index;
      unsigned readers = 0;
      bool only_selects = true;

      for (unsigned j = n + 1; j < count; j++) {
         for (unsigned i = 0; i < code[j].Instruction.NumSrcRegs; i++) {
            const struct tgsi_src_register *src = &code[j].Src[i].Register;

            if (src->File != TGSI_FILE_TEMPORARY || src->Index != temp)
               continue;
            if (code[j].Instruction.Opcode == TGSI_OPCODE_CMP && i == 0 &&
                src->Negate && src->Absolute)
               readers++;
            else
               only_selects = false;
         }
         if (writes_temp(&code[j], temp))
            break;
      }

      if (!only_selects || !readers)
         continue;

      code[n].Instruction.Opcode = TGSI_OPCODE_SEQ;
      for (unsigned j = n + 1; j < count; j++) {
         if (code[j].Instruction.Opcode == TGSI_OPCODE_CMP &&
             code[j].Src[0].Register.File == TGSI_FILE_TEMPORARY &&
             code[j].Src[0].Register.Index == temp) {
            struct tgsi_full_src_register swap = code[j].Src[1];

            code[j].Src[1] = code[j].Src[2];
            code[j].Src[2] = swap;
         }
         if (writes_temp(&code[j], temp))
            break;
      }
   }

   bool any = false;
   for (unsigned n = 0; n < count; n++)
      any |= code[n].Instruction.Opcode == TGSI_OPCODE_SNE;
   if (!any)
      return;

   unsigned one_comp;
   unsigned one_imm = find_immediate(fp, 1.0f, &one_comp);

   struct util_dynarray out;
   util_dynarray_init(&out, NULL);
   for (unsigned n = 0; n < count; n++) {
      struct tgsi_full_instruction inst = code[n];
      if (inst.Instruction.Opcode != TGSI_OPCODE_SNE) {
         util_dynarray_append_typed(&out, struct tgsi_full_instruction, inst);
         continue;
      }
      assert(inst.Dst[0].Register.File == TGSI_FILE_TEMPORARY);
      struct tgsi_full_instruction seq = inst;
      seq.Instruction.Opcode = TGSI_OPCODE_SEQ;
      util_dynarray_append_typed(&out, struct tgsi_full_instruction, seq);

      struct tgsi_full_instruction add = {0};
      add.Instruction.Opcode = TGSI_OPCODE_ADD;
      add.Instruction.NumDstRegs = 1;
      add.Instruction.NumSrcRegs = 2;
      add.Instruction.Saturate = inst.Instruction.Saturate;
      add.Dst[0] = inst.Dst[0];
      add.Src[0].Register.File = TGSI_FILE_IMMEDIATE;
      add.Src[0].Register.Index = one_imm;
      add.Src[0].Register.SwizzleX = add.Src[0].Register.SwizzleY =
         add.Src[0].Register.SwizzleZ = add.Src[0].Register.SwizzleW =
            one_comp;
      add.Src[1].Register.File = TGSI_FILE_TEMPORARY;
      add.Src[1].Register.Index = inst.Dst[0].Register.Index;
      add.Src[1].Register.Negate = 1;
      add.Src[1].Register.SwizzleX = TGSI_SWIZZLE_X;
      add.Src[1].Register.SwizzleY = TGSI_SWIZZLE_Y;
      add.Src[1].Register.SwizzleZ = TGSI_SWIZZLE_Z;
      add.Src[1].Register.SwizzleW = TGSI_SWIZZLE_W;
      util_dynarray_append_typed(&out, struct tgsi_full_instruction, add);
   }
   util_dynarray_fini(instructions);
   *instructions = out;
}

/* Drop the instructions whose results nothing reads, and the channels of
 * partly dead results nothing reads. The passes of dual-source blending
 * store one of the shader's two outputs, which leaves the computation of
 * the other behind. Returns the new count. */
static unsigned
eliminate_dead_code(struct tgsi_full_instruction *code, unsigned count)
{
   uint8_t live[GRATE_FP_MAX_TEMPS + 1] = {0};
   bool *keep = calloc(count, sizeof(*keep));
   unsigned kept = 0;

   assert(keep);
   for (int n = (int)count - 1; n >= 0; n--) {
      struct tgsi_full_instruction *inst = &code[n];
      bool needed = inst->Instruction.NumDstRegs == 0;

      for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
         const struct tgsi_dst_register *dst = &inst->Dst[d].Register;

         if (dst->File != TGSI_FILE_TEMPORARY ||
             (dst->WriteMask & live[dst->Index]))
            needed = true;
      }
      if (!needed)
         continue;

      keep[n] = true;
      kept++;
      for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
         struct tgsi_dst_register *dst = &inst->Dst[d].Register;

         if (dst->File != TGSI_FILE_TEMPORARY)
            continue;
         /* A fetch always writes all four channels */
         if (inst->Instruction.Opcode != TGSI_OPCODE_TEX)
            dst->WriteMask &= live[dst->Index];
         live[dst->Index] &= ~dst->WriteMask;
      }
      for (unsigned s = 0; s < inst->Instruction.NumSrcRegs; s++) {
         const struct tgsi_src_register *src = &inst->Src[s].Register;

         if (src->File == TGSI_FILE_TEMPORARY)
            live[src->Index] |= tgsi_util_get_inst_usage_mask(inst, s);
      }
   }

   unsigned out = 0;
   for (unsigned n = 0; n < count; n++) {
      if (keep[n])
         code[out++] = code[n];
   }
   free(keep);
   assert(out == kept);
   return out;
}

/* NIR emits "MOV t.xy, IN[k].xy; TEX t, t.xy" when the sample overwrites t.
 * The sampler can read a varying directly (the case emit_TEX() handles best),
 * so sample IN[k] and drop the copy, provided the TEX rewrites the copied
 * channels. Returns the new count. */
static unsigned
forward_tex_coord(struct tgsi_full_instruction *code, unsigned count)
{
   for (unsigned n = 1; n < count; n++) {
      struct tgsi_full_instruction *tex = &code[n];
      struct tgsi_full_instruction *mov = &code[n - 1];
      if (tex->Instruction.Opcode != TGSI_OPCODE_TEX ||
          mov->Instruction.Opcode != TGSI_OPCODE_MOV)
         continue;
      const struct tgsi_src_register *coord = &tex->Src[0].Register;
      const struct tgsi_src_register *from = &mov->Src[0].Register;
      if (coord->File != TGSI_FILE_TEMPORARY ||
          mov->Dst[0].Register.File != TGSI_FILE_TEMPORARY ||
          mov->Dst[0].Register.Index != coord->Index ||
          (mov->Dst[0].Register.WriteMask & 3) != 3 ||
          coord->SwizzleX != TGSI_SWIZZLE_X ||
          coord->SwizzleY != TGSI_SWIZZLE_Y || coord->Negate ||
          coord->Absolute)
         continue;
      if (from->File != TGSI_FILE_INPUT || from->SwizzleX != TGSI_SWIZZLE_X ||
          from->SwizzleY != TGSI_SWIZZLE_Y || from->Negate ||
          from->Absolute || mov->Instruction.Saturate)
         continue;
      /* The channels that the copy wrote beyond xy must not be read after it
       * either, and the TEX must replace them. */
      const struct tgsi_dst_register *td = &tex->Dst[0].Register;
      if (td->File != TGSI_FILE_TEMPORARY || td->Index != coord->Index ||
          (mov->Dst[0].Register.WriteMask & ~td->WriteMask))
         continue;
      tex->Src[0].Register = *from;
      memmove(&code[n - 1], &code[n], (count - n) * sizeof(*code));
      count--;
      n--;
   }
   return count;
}

/* Append the instructions for dual-source output modes (see
 * enum grate_fp_output_mode). t0 and t1 are the temporaries that hold
 * output 0 and 1. Returns the temporary to store as the colour. */
static bool
build_dual_output(struct grate_fp_shader *fp,
                  struct util_dynarray *instructions,
                  const struct grate_fp_output *output,
                  const int *output_temps, const unsigned *output_masks,
                  int *result)
{
   if (output_temps[1] < 0) {
      fprintf(stderr, "GRATE FRAG: dual-source blending but the program "
                      "has no second output\n");
      fp->allocation_failed = true;
      return false;
   }

   if (output->mode == GRATE_FP_OUTPUT_SRC1) {
      *result = output_temps[1];
      return true;
   }

   assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
   unsigned factor_temp = fp->num_temporaries++;
   unsigned one_comp, zero_comp;
   unsigned one_imm = find_immediate(fp, 1.0f, &one_comp);
   unsigned zero_imm = find_immediate(fp, 0.0f, &zero_comp);

   /* group 0: the colour channels, group 1: alpha */
   for (unsigned group = 0; group < 2; group++) {
      enum grate_fp_dual_factor factor = output->factor[group];
      unsigned mask = group ? TGSI_WRITEMASK_W : TGSI_WRITEMASK_XYZ;
      struct tgsi_full_instruction mov = {0};

      mov.Instruction.Opcode = TGSI_OPCODE_MOV;
      mov.Instruction.NumDstRegs = 1;
      mov.Instruction.NumSrcRegs = 1;
      mov.Dst[0].Register.File = TGSI_FILE_TEMPORARY;
      mov.Dst[0].Register.Index = factor_temp;
      mov.Dst[0].Register.WriteMask = mask;

      bool invert = factor == GRATE_FP_DUAL_INV_SRC1_COLOR ||
                    factor == GRATE_FP_DUAL_INV_SRC1_ALPHA;
      if (factor == GRATE_FP_DUAL_ZERO || factor == GRATE_FP_DUAL_ONE) {
         unsigned comp = factor == GRATE_FP_DUAL_ONE ? one_comp : zero_comp;

         mov.Src[0].Register.File = TGSI_FILE_IMMEDIATE;
         mov.Src[0].Register.Index =
            factor == GRATE_FP_DUAL_ONE ? one_imm : zero_imm;
         set_src_swizzle(&mov.Src[0].Register, comp, comp, comp, comp);
      } else {
         mov.Src[0].Register.File = TGSI_FILE_TEMPORARY;
         mov.Src[0].Register.Index = output_temps[1];
         if (factor == GRATE_FP_DUAL_SRC1_COLOR ||
             factor == GRATE_FP_DUAL_INV_SRC1_COLOR)
            set_src_swizzle(&mov.Src[0].Register, TGSI_SWIZZLE_X,
                            TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W);
         else
            set_src_swizzle(&mov.Src[0].Register, TGSI_SWIZZLE_W,
                            TGSI_SWIZZLE_W, TGSI_SWIZZLE_W, TGSI_SWIZZLE_W);
      }
      util_dynarray_append_typed(instructions, struct tgsi_full_instruction,
                                 mov);

      if (invert) {
         /* factor = 1 - factor */
         struct tgsi_full_instruction add = {0};

         add.Instruction.Opcode = TGSI_OPCODE_ADD;
         add.Instruction.NumDstRegs = 1;
         add.Instruction.NumSrcRegs = 2;
         add.Dst[0] = mov.Dst[0];
         add.Src[0].Register.File = TGSI_FILE_IMMEDIATE;
         add.Src[0].Register.Index = one_imm;
         set_src_swizzle(&add.Src[0].Register, one_comp, one_comp, one_comp,
                         one_comp);
         add.Src[1].Register.File = TGSI_FILE_TEMPORARY;
         add.Src[1].Register.Index = factor_temp;
         add.Src[1].Register.Negate = 1;
         set_src_swizzle(&add.Src[1].Register, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y,
                         TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W);
         util_dynarray_append_typed(instructions,
                                    struct tgsi_full_instruction, add);
      }
   }

   /* factor = src0 * factor */
   struct tgsi_full_instruction mul = {0};

   mul.Instruction.Opcode = TGSI_OPCODE_MUL;
   mul.Instruction.NumDstRegs = 1;
   mul.Instruction.NumSrcRegs = 2;
   mul.Dst[0].Register.File = TGSI_FILE_TEMPORARY;
   mul.Dst[0].Register.Index = factor_temp;
   mul.Dst[0].Register.WriteMask = TGSI_WRITEMASK_XYZW;
   mul.Src[0].Register.File = TGSI_FILE_TEMPORARY;
   mul.Src[0].Register.Index = output_temps[0];
   mul.Src[1].Register.File = TGSI_FILE_TEMPORARY;
   mul.Src[1].Register.Index = factor_temp;
   for (unsigned i = 0; i < 2; i++) {
      set_src_swizzle(&mul.Src[i].Register, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y,
                      TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W);
   }
   util_dynarray_append_typed(instructions, struct tgsi_full_instruction,
                              mul);

   *result = factor_temp;
   return true;
}

/* The spill allocator pins the operands of a whole instruction. Splitting
 * independent lanes lets a three-source vector operation run without
 * keeping all twelve source scalars resident simultaneously. */
static void
scalarize_for_spilling(struct util_dynarray *instructions)
{
   struct util_dynarray out;
   util_dynarray_init(&out, NULL);
   util_dynarray_foreach (instructions, struct tgsi_full_instruction, code) {
      bool split = code->Instruction.NumDstRegs == 1;
      switch (code->Instruction.Opcode) {
      case TGSI_OPCODE_MOV:
      case TGSI_OPCODE_ADD:
      case TGSI_OPCODE_MUL:
      case TGSI_OPCODE_MIN:
      case TGSI_OPCODE_SLT:
      case TGSI_OPCODE_SGE:
      case TGSI_OPCODE_SEQ:
      case TGSI_OPCODE_CMP:
      case TGSI_OPCODE_LRP:
         break;
      default:
         split = false;
         break;
      }
      if (!split) {
         util_dynarray_append_typed(&out, struct tgsi_full_instruction,
                                    *code);
         continue;
      }
      const struct tgsi_dst_register *dst = &code->Dst[0].Register;
      split &= dst->File == TGSI_FILE_TEMPORARY && !dst->Indirect;
      for (unsigned s = 0; s < code->Instruction.NumSrcRegs; s++) {
         const struct tgsi_src_register *src = &code->Src[s].Register;
         split &= !src->Indirect;
         if (src->File != TGSI_FILE_TEMPORARY || src->Index != dst->Index)
            continue;
         /* A vector instruction reads every source before writing. Leave
          * it intact if an earlier scalar write would destroy a source
          * that a later lane still needs, such as MOV t.xy, t.yx. */
         for (unsigned lane = 0; lane < 4; lane++) {
            unsigned component = src_component(src, lane);
            if ((dst->WriteMask & (1 << lane)) && component < lane &&
                (dst->WriteMask & (1 << component)))
               split = false;
         }
      }
      if (!split) {
         util_dynarray_append_typed(&out, struct tgsi_full_instruction,
                                    *code);
         continue;
      }
      for (unsigned lane = 0; lane < 4; lane++) {
         if (!(dst->WriteMask & (1 << lane)))
            continue;
         struct tgsi_full_instruction scalar = *code;
         scalar.Dst[0].Register.WriteMask = 1 << lane;
         util_dynarray_append_typed(&out, struct tgsi_full_instruction,
                                    scalar);
      }
   }
   util_dynarray_fini(instructions);
   *instructions = out;
}

/* NVIDIA selects on the sign of an FP20 difference. Avoid rounding that
 * difference through a boolean predicate when SLT only feeds selections. */
static void
lower_select_comparisons(struct util_dynarray *instructions)
{
   struct tgsi_full_instruction *code = instructions->data;
   unsigned count =
      util_dynarray_num_elements(instructions, struct tgsi_full_instruction);

   for (unsigned n = 0; n < count; n++) {
      struct tgsi_full_instruction *cmp = &code[n];
      struct tgsi_dst_register *dst = &cmp->Dst[0].Register;
      if (cmp->Instruction.Opcode != TGSI_OPCODE_SLT ||
          cmp->Instruction.Saturate || dst->File != TGSI_FILE_TEMPORARY ||
          !util_is_power_of_two_nonzero(dst->WriteMask))
         continue;
      unsigned component = util_logbase2(dst->WriteMask);
      bool valid = true, used = false;
      unsigned end = count;
      for (unsigned m = n + 1; m < count && valid; m++) {
         struct tgsi_full_instruction *inst = &code[m];
         for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
            const struct tgsi_src_register *src = &inst->Src[i].Register;
            if (src->File != TGSI_FILE_TEMPORARY || src->Index != dst->Index)
               continue;
            used = true;
            valid &= inst->Instruction.Opcode == TGSI_OPCODE_CMP && i == 0 &&
                     src->Negate && !src->Indirect && !src->Dimension &&
                     src->SwizzleX == component &&
                     src->SwizzleY == component &&
                     src->SwizzleZ == component && src->SwizzleW == component;
         }
         for (unsigned i = 0; i < inst->Instruction.NumDstRegs; i++) {
            const struct tgsi_dst_register *write = &inst->Dst[i].Register;
            if (write->File == dst->File && write->Index == dst->Index &&
                (write->WriteMask & dst->WriteMask)) {
               end = m + 1;
               m = count;
               break;
            }
         }
      }
      if (!valid || !used)
         continue;
      /* SLT(a,b) is true exactly when a-b is negative. */
      cmp->Instruction.Opcode = TGSI_OPCODE_ADD;
      cmp->Src[1].Register.Negate = !cmp->Src[1].Register.Negate;
      for (unsigned m = n + 1; m < end; m++) {
         struct tgsi_src_register *src = &code[m].Src[0].Register;
         if (code[m].Instruction.Opcode == TGSI_OPCODE_CMP &&
             src->File == dst->File && src->Index == dst->Index) {
            src->Negate = false;
            src->Absolute = false;
         }
      }
   }
}

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi,
                 const struct grate_fp_output *output, bool spilling)
{
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);

   fp->info.num_inputs = 0;
   fp->info.num_constants = 0;
   fp->info.num_spill_slots = 0;
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;
   fp->num_immediates = 0;
   fp->num_temporaries = 0;
   fp->allocation_failed = false;
   fp->spilling = spilling;
   fp->uses_derivatives = false;
   fp->avoid_rows = false;
   fp->register_pressure = false;
   fp->current_instruction = NULL;
   fp->code_end = NULL;
   memset(fp->pinned, 0, sizeof(fp->pinned));
   memset(fp->spill_used, 0, sizeof(fp->spill_used));
   memset(fp->spill_fx10, 0, sizeof(fp->spill_fx10));
   memset(fp->occupied, 0, sizeof(fp->occupied));
   memset(fp->temp_src, -1, sizeof(fp->temp_src));
   memset(fp->temp_dst, -1, sizeof(fp->temp_dst));
   memset(fp->spill_slot, -1, sizeof(fp->spill_slot));
   struct util_dynarray instructions;
   util_dynarray_init(&instructions, NULL);

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_IMMEDIATE: {
         const struct tgsi_full_immediate *imm =
            &tgsi->FullToken.FullImmediate;
         unsigned n =
            imm->Immediate.NrTokens - 1; /* number of 32-bit values: 1 to 4 */

         assert(fp->num_immediates < GRATE_FP_MAX_IMMEDIATES);
         for (unsigned i = 0; i < 4; ++i)
            fp->immediates[fp->num_immediates][i] =
               i < n ? imm->u[i].Float : 0.0f;
         fp->num_immediates++;
         break;
      }

      case TGSI_TOKEN_TYPE_DECLARATION:
         emit_tgsi_declaration(fp, &tgsi->FullToken.FullDeclaration);
         break;

      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode != TGSI_OPCODE_END)
            util_dynarray_append_typed(&instructions,
                                       struct tgsi_full_instruction,
                                       tgsi->FullToken.FullInstruction);
         break;
      }
   }

   util_dynarray_foreach (&instructions, struct tgsi_full_instruction,
                          derivative) {
      fp->uses_derivatives |=
         derivative->Instruction.Opcode == TGSI_OPCODE_DDX ||
         derivative->Instruction.Opcode == TGSI_OPCODE_DDY;
   }

   if (emit_direct_varying_product(
          fp, instructions.data,
          util_dynarray_num_elements(&instructions,
                                     struct tgsi_full_instruction))) {
      util_dynarray_fini(&instructions);
      return;
   }

   lower_sne(fp, &instructions);
   lower_select_comparisons(&instructions);

   /* Fragment output values stay virtual until all shader operations have
    * consumed the sampled pixel. Early masked DW stores alias R2/R3 and
    * would destroy still-live texture channels. */
   int output_temps[16];
   unsigned output_masks[16] = {0};
   memset(output_temps, -1, sizeof(output_temps));
   util_dynarray_foreach (&instructions, struct tgsi_full_instruction,
                          instruction) {
      for (unsigned d = 0; d < instruction->Instruction.NumDstRegs; d++) {
         struct tgsi_dst_register *dst = &instruction->Dst[d].Register;
         if (dst->File != TGSI_FILE_OUTPUT)
            continue;
         unsigned index = dst->Index;
         assert(index < 16);
         if (output_temps[index] < 0)
            output_temps[index] = fp->num_temporaries++;
         output_masks[index] |= dst->WriteMask;
         dst->File = TGSI_FILE_TEMPORARY;
         dst->Index = output_temps[index];
      }
   }
   /* One colour is stored: output 0, or what a dual-source pass makes of
    * the two outputs. */
   int stored_temp = output_temps[0];
   if (output->mode != GRATE_FP_OUTPUT_SRC0 &&
       !build_dual_output(fp, &instructions, output, output_temps,
                          output_masks, &stored_temp)) {
      util_dynarray_fini(&instructions);
      return;
   }
   if (stored_temp >= 0) {
      struct tgsi_full_instruction store = {0};

      store.Instruction.Opcode = TGSI_OPCODE_MOV;
      store.Instruction.NumDstRegs = 1;
      store.Instruction.NumSrcRegs = 1;
      store.Dst[0].Register.File = TGSI_FILE_OUTPUT;
      store.Dst[0].Register.Index = 0;
      store.Dst[0].Register.WriteMask = output->mode == GRATE_FP_OUTPUT_SRC1
                                           ? output_masks[1]
                                           : output_masks[0];
      store.Src[0].Register.File = TGSI_FILE_TEMPORARY;
      store.Src[0].Register.Index = stored_temp;
      set_src_swizzle(&store.Src[0].Register, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y,
                      TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W);
      util_dynarray_append_typed(&instructions, struct tgsi_full_instruction,
                                 store);
   }
   assert(fp->num_temporaries < GRATE_FP_MAX_TEMPS);
   if (fp->spilling)
      scalarize_for_spilling(&instructions);
   unsigned count =
      util_dynarray_num_elements(&instructions, struct tgsi_full_instruction);
   struct tgsi_full_instruction *code = instructions.data;
   count = forward_tex_coord(code, count);
   count = eliminate_dead_code(code, count);
   fp->code_end = code + count;
   uint8_t (*live_after)[GRATE_FP_MAX_TEMPS] =
      calloc(count + 1, sizeof(*live_after));
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
            live_after[n][reg->Index] |=
               tgsi_util_get_inst_usage_mask(inst, src);
      }
   }
   /* Rows each instruction overwrites, and where the last sampled computed
    * coordinate is: before that fetch values have to live in rows. */
   uint8_t *clobber = calloc(count + 1, 1);
   uint8_t *clobber_after = calloc(count + 1, 1);
   int row_limit = -1;
   bool has_fetch = false;
   assert(clobber && clobber_after);
   for (unsigned n = 0; n < count; n++)
      has_fetch |= code[n].Instruction.Opcode == TGSI_OPCODE_TEX;
   for (unsigned n = 0; n < count; n++) {
      const struct tgsi_full_instruction *inst = &code[n];
      unsigned write_mask =
         inst->Instruction.NumDstRegs ? inst->Dst[0].Register.WriteMask : 0;
      for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++) {
         const struct tgsi_src_register *reg = &inst->Src[src].Register;
         if (reg->File != TGSI_FILE_INPUT)
            continue;
         unsigned swizzle[] = {reg->SwizzleX, reg->SwizzleY, reg->SwizzleZ,
                               reg->SwizzleW};
         for (unsigned c = 0; c < 4; c++)
            if (inst->Instruction.Opcode == TGSI_OPCODE_TEX
                   ? c < 2
                   : (write_mask & (1 << c)))
               clobber[n] |= 1 << swizzle[c];
      }
      if (inst->Instruction.Opcode == TGSI_OPCODE_TEX) {
         clobber_after[n] |= 0xc;
         if (inst->Src[0].Register.File != TGSI_FILE_INPUT)
            clobber_after[n] |= 0x3;
      }
      /* In a program with a texture fetch, a value that is alive across a
       * fetch or an interpolation has to be in a row. */
      if (has_fetch &&
          (inst->Instruction.Opcode == TGSI_OPCODE_TEX || reads_input(inst)))
         row_limit = n;
   }
   for (unsigned n = 0; n < count; n++) {
      const struct tgsi_full_instruction *inst = &code[n];
      bool single_input = true;
      fp->current_instruction = inst;
      int input = -1;
      for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++) {
         const struct tgsi_src_register *reg = &inst->Src[src].Register;
         if (reg->File != TGSI_FILE_INPUT)
            continue;
         if (input >= 0 && input != reg->Index)
            single_input = false;
         input = reg->Index;
      }
      fp->late_rows =
         (int)n > last_fetch || ((int)n == last_fetch && single_input);
      fp->texture_rows =
         (int)n == last_fetch && inst->Instruction.Opcode == TGSI_OPCODE_TEX;
      memset(fp->claimed, 0, sizeof(fp->claimed));
      memset(fp->reusable, 0, sizeof(fp->reusable));
      memcpy(fp->temp_dst, fp->temp_src, sizeof(fp->temp_dst));
      memset(fp->temp_dst[fp->num_temporaries], -1, sizeof(fp->temp_dst[0]));
      if (fp->spilling) {
         memset(fp->pinned, 0, sizeof(fp->pinned));
         /* Pin every resident source before loading any spilled source.
          * Swizzles and partial opcode reads are included in this mask. */
         for (unsigned s = 0; s < inst->Instruction.NumSrcRegs; s++) {
            const struct tgsi_src_register *src = &inst->Src[s].Register;
            if (src->File != TGSI_FILE_TEMPORARY)
               continue;
            unsigned used = tgsi_util_get_inst_usage_mask(inst, s);
            for (unsigned c = 0; c < 4; c++) {
               int loc = fp->temp_src[src->Index][c];
               if ((used & (1 << c)) && loc >= 0 && loc != 31)
                  fp->pinned[loc & 0x7f] |= scalar_mask(loc);
            }
         }
         for (unsigned s = 0; s < inst->Instruction.NumSrcRegs; s++) {
            const struct tgsi_src_register *src = &inst->Src[s].Register;
            if (src->File != TGSI_FILE_TEMPORARY)
               continue;
            unsigned used = tgsi_util_get_inst_usage_mask(inst, s);
            for (unsigned c = 0; c < 4; c++) {
               int old = fp->temp_src[src->Index][c];
               if (!(used & (1 << c)) || old >= -1)
                  continue;
               unsigned slot = -2 - old;
               /* The current instruction may still interpolate or fetch
                * into R1-R3 before consuming the reloaded source. */
               bool avoid_rows = fp->avoid_rows;
               fp->avoid_rows = true;
               int loc = allocate_scalar(fp, fp->spill_fx10[slot] ? 1 : 0);
               fp->avoid_rows = avoid_rows;
               if (fp->allocation_failed)
                  break;
               emit_reload_scalar(fp, loc, 12 | (slot << 6));
               fp->temp_src[src->Index][c] = loc;
               fp->temp_dst[src->Index][c] = loc;
               fp->pinned[loc & 0x7f] |= scalar_mask(loc);
            }
         }
         /* Loaded inputs have the same reuse opportunities as inputs that
          * were resident already. Output allocation claims them later. */
         memset(fp->claimed, 0, sizeof(fp->claimed));
         if (fp->allocation_failed)
            break;
      }
      bool predicate = inst->Instruction.Opcode == TGSI_OPCODE_SLT ||
                       inst->Instruction.Opcode == TGSI_OPCODE_SGE ||
                       inst->Instruction.Opcode == TGSI_OPCODE_SEQ;
      for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
         const struct tgsi_dst_register *dst = &inst->Dst[d].Register;
         if (dst->File != TGSI_FILE_TEMPORARY)
            continue;
         for (unsigned c = 0; c < 4; c++) {
            if (!(dst->WriteMask & (1 << c)))
               continue;
            unsigned kind = predicate                                     ? 2
                            : inst->Instruction.Opcode == TGSI_OPCODE_TEX ? 1
                                                                          : 0;
            if (inst->Instruction.Opcode == TGSI_OPCODE_MOV ||
                inst->Instruction.Opcode == TGSI_OPCODE_CMP) {
               bool packed = true;
               unsigned first =
                  inst->Instruction.Opcode == TGSI_OPCODE_CMP ? 1 : 0;
               for (unsigned src = first; src < inst->Instruction.NumSrcRegs;
                    src++) {
                  const struct tgsi_src_register *operand =
                     &inst->Src[src].Register;
                  unsigned swizzle[] = {operand->SwizzleX, operand->SwizzleY,
                                        operand->SwizzleZ, operand->SwizzleW};
                  packed &=
                     operand->File == TGSI_FILE_TEMPORARY &&
                     fp->temp_src[operand->Index][swizzle[c]] >= 0 &&
                     (fp->temp_src[operand->Index][swizzle[c]] & TEMP_FX10);
               }
               if (packed)
                  kind = 1;
            }
            fp->avoid_rows = inst->Instruction.Opcode == TGSI_OPCODE_FRC ||
                             inst->Instruction.Opcode == TGSI_OPCODE_FLR;
            for (unsigned m = n + 1; m < count && !fp->avoid_rows; m++) {
               switch (code[m].Instruction.Opcode) {
               case TGSI_OPCODE_FRC:
               case TGSI_OPCODE_FLR:
                  /* Leave row scratch available even for a dying source. */
                  fp->avoid_rows = live_after[m][dst->Index] & (1 << c);
                  break;
               case TGSI_OPCODE_RCP:
               case TGSI_OPCODE_RSQ:
               case TGSI_OPCODE_SQRT:
               case TGSI_OPCODE_LG2:
               case TGSI_OPCODE_EX2:
               case TGSI_OPCODE_POW:
                  fp->avoid_rows =
                     live_after[m + 1][dst->Index] & (1 << c);
                  break;
               default:
                  break;
               }
            }
            memset(fp->reusable, 0, sizeof(fp->reusable));
            bool simple = inst->Instruction.Opcode != TGSI_OPCODE_TEX;
            for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++)
               simple &= inst->Src[src].Register.File != TGSI_FILE_INPUT;
            if (simple) {
               for (unsigned reg = 0; reg < 16; reg++) {
                  unsigned needed = 0;
                  for (unsigned t = 0; t < fp->num_temporaries; t++)
                     for (unsigned k = 0; k < 4; k++) {
                        int loc = fp->temp_src[t][k];
                        if (loc < 0 || (loc & 0x7f) != reg)
                           continue;
                        unsigned mask = (loc & TEMP_FX10)
                                           ? ((loc & TEMP_HIGH) ? 2 : 1)
                                           : 3;
                        bool overwritten = false;
                        for (unsigned d2 = 0;
                             d2 < inst->Instruction.NumDstRegs; d2++) {
                           const struct tgsi_dst_register *out =
                              &inst->Dst[d2].Register;
                           overwritten |= out->File == TGSI_FILE_TEMPORARY &&
                                          out->Index == t &&
                                          (out->WriteMask & (1 << k));
                        }
                        if (!overwritten && (live_after[n + 1][t] & (1 << k)))
                           needed |= mask;
                     }
                  for (unsigned src = 0; src < inst->Instruction.NumSrcRegs;
                       src++) {
                     const struct tgsi_src_register *operand =
                        &inst->Src[src].Register;
                     if (operand->File != TGSI_FILE_TEMPORARY)
                        continue;
                     unsigned swizzle[] = {
                        operand->SwizzleX, operand->SwizzleY,
                        operand->SwizzleZ, operand->SwizzleW};
                     for (unsigned lane = c + 1; lane < 4; lane++) {
                        int loc = fp->temp_src[operand->Index][swizzle[lane]];
                        if ((dst->WriteMask & (1 << lane)) && loc >= 0 &&
                            (loc & 0x7f) == reg)
                           needed |= (loc & TEMP_FX10)
                                        ? ((loc & TEMP_HIGH) ? 2 : 1)
                                        : 3;
                     }
                  }
                  fp->reusable[reg] = (~needed) & 3;
               }
            }
            fp->temp_dst[dst->Index][c] =
               (live_after[n + 1][dst->Index] & (1 << c))
                  ? (fp->texture_rows
                        ? (2 + c / 2) | TEMP_FX10 | (c % 2 ? TEMP_HIGH : 0)
                        : across_fetch(
                             fp, clobber, clobber_after, live_after, count, n,
                             dst->Index, c, kind,
                             inst->Instruction.Opcode == TGSI_OPCODE_TEX,
                             (int)n < row_limit))
                  : 31;
         }
      }
      if (grate_debug & GRATE_DEBUG_TGSI) {
         fprintf(stderr, "ALLOC %u opcode %u", n, inst->Instruction.Opcode);
         for (unsigned src = 0; src < inst->Instruction.NumSrcRegs; src++) {
            const struct tgsi_src_register *operand =
               &inst->Src[src].Register;
            if (operand->File == TGSI_FILE_TEMPORARY)
               fprintf(stderr, " src%u[%d,%d,%d,%d]", src,
                       fp->temp_src[operand->Index][0],
                       fp->temp_src[operand->Index][1],
                       fp->temp_src[operand->Index][2],
                       fp->temp_src[operand->Index][3]);
         }
         for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
            const struct tgsi_dst_register *out = &inst->Dst[d].Register;
            if (out->File == TGSI_FILE_TEMPORARY)
               fprintf(
                  stderr, " dst[%d,%d,%d,%d]", fp->temp_dst[out->Index][0],
                  fp->temp_dst[out->Index][1], fp->temp_dst[out->Index][2],
                  fp->temp_dst[out->Index][3]);
         }
         fprintf(stderr, "\n");
      }
      if (fp->allocation_failed)
         break;
      emit_tgsi_instr(fp, inst);
      if (fp->allocation_failed)
         break;
      memset(fp->occupied, 0, sizeof(fp->occupied));
      for (unsigned t = 0; t < fp->num_temporaries; t++) {
         for (unsigned c = 0; c < 4; c++) {
            int reg = fp->temp_dst[t][c];
            if (fp->spilling && fp->spill_slot[t][c] >= 0) {
               bool written = false;
               for (unsigned d = 0; d < inst->Instruction.NumDstRegs; d++) {
                  const struct tgsi_dst_register *dst =
                     &inst->Dst[d].Register;
                  written |= dst->File == TGSI_FILE_TEMPORARY &&
                             dst->Index == t && (dst->WriteMask & (1 << c));
               }
               if (written || !(live_after[n + 1][t] & (1 << c))) {
                  fp->spill_used[fp->spill_slot[t][c]] = false;
                  fp->spill_slot[t][c] = -1;
               }
            }
            if (live_after[n + 1][t] & (1 << c)) {
               assert(reg >= 0 || (fp->spilling && reg < -1) ||
                      fp->allocation_failed);
               fp->temp_src[t][c] = reg;
               if (reg >= 0 && reg != 31) {
                  unsigned mask =
                     (reg & TEMP_FX10) ? ((reg & TEMP_HIGH) ? 2 : 1) : 3;
                  fp->occupied[reg & 0x7f] |= mask;
               }
            } else {
               fp->temp_src[t][c] = -1;
            }
         }
      }
      memset(fp->temp_src[fp->num_temporaries], -1, sizeof(fp->temp_src[0]));
   }
   fp->current_instruction = NULL;
   fp->code_end = NULL;
   free(clobber_after);
   free(clobber);
   free(live_after);
   util_dynarray_fini(&instructions);
}
