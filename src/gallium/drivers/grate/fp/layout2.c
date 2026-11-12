/*
 * A fragment program layout like NVIDIA's: a list scheduler over single ALU
 * slot operations and unit operations (interpolation or SFU, fetch, output
 * write) that puts a unit operation on any step (it works on the register row
 * step % rows), allocates the registers while it schedules, and packs several
 * operations into one ALU packet.
 *
 * It takes the program the emitter wrote (registers allocated for units on R0
 * to R3 of row 0, one TGSI instruction per step), splits it into values (one
 * value per register write), and schedules the operations again. The result
 * replaces align_to_rows() only if grate_fp_layout_signature() says it does the
 * same as the original; otherwise the caller keeps the old layout.
 *
 * Rules (docs/grate-alu-latency.md): a unit operation on step s uses the row
 * s % rows and acts before the ALU packets of the step; an ALU result is
 * visible two packets later or in a later round; a round is `rows` steps; the
 * first step of every round holds an ALU packet; a step holds at most three
 * packets; the output write sits on the first step of the last round.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/list.h"
#include "util/macros.h"
#include "util/u_memory.h"

#include "fp/fpir.h"
#include "fp/layout.h"
#include "grate_compiler.h"

#define L2_MAX_OPS 512
#define L2_MAX_DEFS 1024
#define L2_MAX_STEPS 64
#define L2_SRCS 4          /* A, B, C and the old value of a conditional destination */

enum l2_kind { L2_ALU, L2_MFU, L2_TEX, L2_DW };

struct l2_def {
   unsigned width;          /* 2: FP20 in a whole register, 1: one half */
   int tie_k, tie_half;     /* row-relative register and half; -1 when free */
   int cls;                 /* class of the unit operations tied to it */
   int writer;
   int uses_left;
   bool written;
   int step;
   int64_t issue;
   int64_t last_read;       /* position of the last read, issue when none */
   int64_t land;            /* position from which a read in its round sees it */
   int64_t start;           /* earliest moment it may change its register */
   int last_reader;         /* operation of the last read */
   int reg, half;           /* assigned location, -1 when none */
   int fixed_reg, fixed_half; /* a pinned comparison register (R64..R71), -1 when none */
   int readers[16];
   unsigned num_readers;
   bool reg_free;           /* every use is a forward: the slot writes R31 */
   bool dead;               /* folded away */
   int interp_key;          /* tram row << 3 | variable, for an interpolated value, else -1 */
};

struct l2_op {
   enum l2_kind kind;
   struct fp_alu_instr alu;
   int src_def[L2_SRCS];
   int fwd[3];           /* slot of the packet before that supplies source k as R24-R27, -1 none */
   int pre[24];          /* operations that must be placed before this one */
   unsigned num_pre;
   int dst_def;
   const struct fp_mfu_instr *mfu;
   uint32_t tex;
   struct fp_dw_instr dw;
   int in[4];
   unsigned num_in;
   int out[4];
   unsigned num_out;
   int cls;
   int orig;
   int height;
   bool placed;
   bool dead;            /* folded away */
   int step, packet, slot;
};

struct l2 {
   unsigned rows;
   struct l2_op ops[L2_MAX_OPS];
   unsigned num_ops;
   struct l2_def defs[L2_MAX_DEFS];
   unsigned num_defs;
   int cls_parent[L2_MAX_OPS];
   int cls_row[L2_MAX_OPS];
   /* schedule: packets per step, and the slot operations of every packet */
   unsigned steps;
   unsigned step_packets[L2_MAX_STEPS];
   int64_t step_first[L2_MAX_STEPS];
   int step_mfu[L2_MAX_STEPS];
   int step_tex[L2_MAX_STEPS];
   int packet_op[L2_MAX_STEPS][3][4];
   int64_t total_packets;
   bool verbose;
   unsigned folded;      /* copies removed */
   bool prop;            /* also propagate copies that cannot be folded */
   /* an operation may only be scheduled when fewer than this many operations
    * that came before it in the program are still waiting */
   unsigned window;
};

static unsigned l2_fold_copies(struct l2 *s);
static unsigned l2_cse_interp(struct l2 *s);

static int
cls_find(struct l2 *s, int a)
{
   while (s->cls_parent[a] != a)
      a = s->cls_parent[a] = s->cls_parent[s->cls_parent[a]];
   return a;
}

static void
cls_union(struct l2 *s, int a, int b)
{
   a = cls_find(s, a);
   b = cls_find(s, b);
   if (a != b)
      s->cls_parent[b] = a;
}

/* ---- extraction ---- */

static int
new_def(struct l2 *s, unsigned width, int writer)
{
   if (s->num_defs >= L2_MAX_DEFS)
      return -1;
   struct l2_def *d = &s->defs[s->num_defs];
   memset(d, 0, sizeof(*d));
   d->width = width;
   d->tie_k = d->tie_half = -1;
   d->cls = -1;
   d->writer = writer;
   d->reg = d->half = -1;
   d->fixed_reg = d->fixed_half = -1;
   d->interp_key = -1;
   return s->num_defs++;
}

static bool
tie(struct l2 *s, int def, int op, int k, int half)
{
   struct l2_def *d = &s->defs[def];
   if (d->tie_k >= 0 && (d->tie_k != k || d->tie_half != half))
      return false;
   d->tie_k = k;
   d->tie_half = half;
   if (d->cls >= 0)
      cls_union(s, d->cls, op);
   else
      d->cls = op;
   return true;
}

/* the definitions the halves of a register hold, -1 when never written */
struct l2_cur {
   int def[48];
};

/* R0-R15, then R64-R71 */
static int
cur_slot(unsigned reg)
{
   if (reg < 16)
      return reg;
   if (reg >= 64 && reg < 72)
      return 16 + (reg - 64);
   return -1;
}

static int
cur_get(const struct l2_cur *c, unsigned reg, unsigned half)
{
   int sl = cur_slot(reg);
   return sl < 0 ? -1 : c->def[sl * 2 + half];
}

static int
cur_fp20(const struct l2_cur *c, unsigned reg)
{
   int a = cur_get(c, reg, 0), b = cur_get(c, reg, 1);
   return a == b ? a : -2;
}

static bool
l2_extract(struct l2 *s, const struct grate_fp_shader *fp)
{
   unsigned num_packets = list_length(&fp->alu_instructions);
   unsigned num_mfus = list_length(&fp->mfu_instructions);
   const struct fp_alu_instr_packet **pk = calloc(num_packets + 1, sizeof(*pk));
   const struct fp_mfu_instr **mf = calloc(num_mfus + 1, sizeof(*mf));
   if (!pk || !mf) {
      free(pk);
      free(mf);
      return false;
   }
   unsigned i = 0;
   list_for_each_entry (struct fp_alu_instr_packet, p, &fp->alu_instructions,
                        link)
      pk[i++] = p;
   i = 0;
   list_for_each_entry (struct fp_mfu_instr, m, &fp->mfu_instructions, link)
      mf[i++] = m;

   struct l2_cur cur;
   memset(&cur, 0xff, sizeof(cur));
   bool ok = true;
   bool have_dw = false;
   /* the value stored under each spill slot, so that a reload can read it */
   struct {
      unsigned key;
      int def;
   } mem[256];
   unsigned num_mem = 0;

#define NEWOP(k_)                                                            \
   ({                                                                        \
      if (s->num_ops >= L2_MAX_OPS) {                                        \
         ok = false;                                                         \
         break;                                                              \
      }                                                                      \
      struct l2_op *o_ = &s->ops[s->num_ops];                                \
      memset(o_, 0, sizeof(*o_));                                            \
      o_->kind = (k_);                                                       \
      o_->orig = s->num_ops;                                                 \
      o_->cls = s->num_ops;                                                  \
      o_->dst_def = -1;                                                      \
      s->cls_parent[s->num_ops] = s->num_ops;                                \
      s->cls_row[s->num_ops] = -1;                                           \
      o_;                                                                    \
   })

   list_for_each_entry (struct fp_instr, instr, &fp->fp_instructions, link) {
      if (instr->pseq && !instr->spill_load) {
         ok = false;
         break;
      }
      int reload_def = -1;
      if (instr->spill_load) {
         unsigned key = (instr->pseq & 0xffff) >> 6;
         for (unsigned k = 0; k < num_mem; k++)
            if (mem[k].key == key)
               reload_def = mem[k].def;
         if (reload_def < 0) {
            ok = false;
            break;
         }
      }
      if (instr->dw.enable && instr->dw.spill) {
         /* a spill store: remember the value, the store itself goes away */
         if (instr->alu_sched.num_instructions != 1 || num_mem >= 256) {
            ok = false;
            break;
         }
         const struct fp_alu_instr *a =
            &pk[instr->alu_sched.address]->slots[0];
         int d;
         if (a->src[0].index >= 16) {
            ok = false;
            break;
         }
         d = a->src[0].datatype == FP_DATATYPE_FIXED10
                ? cur_get(&cur, a->src[0].index, a->src[0].sub_reg_select_high)
                : cur_fp20(&cur, a->src[0].index);
         if (d < 0) {
            ok = false;
            break;
         }
         unsigned key = instr->dw.spill_args >> 4, k;
         for (k = 0; k < num_mem && mem[k].key != key; k++)
            ;
         mem[k].key = key;
         mem[k].def = d;
         if (k == num_mem)
            num_mem++;
         continue;
      }
      if (instr->mfu_sched.num_instructions) {
         const struct fp_mfu_instr *m = mf[instr->mfu_sched.address];
         bool sfu = m->sfu.op != FP_SFU_OP_NOP && m->sfu.reg < 4;
         bool var = false;
         for (unsigned k = 0; k < 4; k++)
            var |= m->var[k].op != FP_VAR_OP_NOP;
         if ((sfu && var) || (!sfu && !var) || (sfu && m->sfu.reg > 3)) {
            /* a step with an interpolation helper only: nothing to move */
            if (!sfu && !var)
               continue;
            ok = false;
            break;
         }
         struct l2_op *op = NEWOP(L2_MFU);
         op->mfu = m;
         int oi = s->num_ops++;
         if (sfu) {
            int din = cur_fp20(&cur, m->sfu.reg);
            if (din < 0) {
               ok = false;
               break;
            }
            op->in[op->num_in++] = din;
            if (!tie(s, din, oi, m->sfu.reg, -1)) {
               ok = false;
               break;
            }
            int dout = new_def(s, 2, oi);
            if (dout < 0 || !tie(s, dout, oi, m->sfu.reg, -1)) {
               ok = false;
               break;
            }
            op->out[op->num_out++] = dout;
            cur.def[m->sfu.reg * 2] = cur.def[m->sfu.reg * 2 + 1] = dout;
         } else {
            for (unsigned k = 0; k < 4; k++) {
               if (m->var[k].op == FP_VAR_OP_NOP)
                  continue;
               if (m->var[k].op != FP_VAR_OP_FP20) {
                  ok = false;
                  break;
               }
               int d = new_def(s, 2, oi);
               if (d < 0 || !tie(s, d, oi, k, -1)) {
                  ok = false;
                  break;
               }
               s->defs[d].interp_key = (m->var[k].tram_row << 3) | k;
               op->out[op->num_out++] = d;
               cur.def[k * 2] = cur.def[k * 2 + 1] = d;
            }
            if (!ok)
               break;
         }
      }
      if (instr->tex & (1 << 10)) {
         if (instr->tex & 0x10) {
            ok = false;
            break;
         }
         struct l2_op *op = NEWOP(L2_TEX);
         op->tex = instr->tex;
         int oi = s->num_ops++;
         for (unsigned k = 0; k < 2; k++) {
            int d = cur_fp20(&cur, k);
            if (d < 0 || !tie(s, d, oi, k, -1)) {
               ok = false;
               break;
            }
            op->in[op->num_in++] = d;
         }
         if (!ok)
            break;
         for (unsigned c = 0; c < 4; c++) {
            int d = new_def(s, 1, oi);
            if (d < 0 || !tie(s, d, oi, 2 + c / 2, c % 2)) {
               ok = false;
               break;
            }
            op->out[op->num_out++] = d;
            cur.def[(2 + c / 2) * 2 + c % 2] = d;
         }
         if (!ok)
            break;
      }
      for (unsigned p = 0; p < instr->alu_sched.num_instructions && ok; p++) {
         unsigned address = instr->alu_sched.address + p;
         if (address >= num_packets) {
            ok = false;
            break;
         }
         struct pend {
            int op;
            unsigned reg, half_lo, half_hi;
            int def;
         } pending[4];
         unsigned npend = 0;
         for (unsigned sl = 0; sl < 4 && ok; sl++) {
            const struct fp_alu_instr *a = &pk[address]->slots[sl];
            if (!a->dst.write_low_sub_reg && !a->dst.write_high_sub_reg)
               continue;
            bool pinned = a->dst.index >= 64 && a->dst.index < 72;
            if ((a->dst.index >= 16 && a->dst.index != 31 && !pinned) ||
                (a->condition != FP_CONDITION_ALWAYS && !pinned) ||
                (pinned && a->dst.write_low_sub_reg == a->dst.write_high_sub_reg)) {
               ok = false;
               break;
            }
            struct l2_op *op = NEWOP(L2_ALU);
            op->alu = *a;
            int oi = s->num_ops++;
            op->cls = -1;
            op->src_def[3] = -1;
            for (unsigned k = 0; k < 3; k++) {
               const struct fp_alu_src_operand *src = &a->src[k];
               op->src_def[k] = -1;
               if (src->index >= 24 && src->index <= 27) {
                  ok = false;
                  break;
               }
               if (reload_def >= 0 && src->index == 12) {
                  op->src_def[k] = reload_def;
                  continue;
               }
               if (src->index >= 16 && cur_slot(src->index) < 0)
                  continue;
               int d;
               if (src->datatype == FP_DATATYPE_FIXED10) {
                  d = cur_get(&cur, src->index, src->sub_reg_select_high);
               } else {
                  d = cur_fp20(&cur, src->index);
                  if (d >= 0 && s->defs[d].width != 2)
                     d = -2;
               }
               if (d < 0) {
                  ok = false;
                  break;
               }
               op->src_def[k] = d;
            }
            if (!ok)
               break;
            /* a conditional write keeps the old value when it does not
             * happen, so the old value is an operand */
            if (a->condition != FP_CONDITION_ALWAYS) {
               int old = cur_get(&cur, a->dst.index,
                                 a->dst.write_low_sub_reg ? 0 : 1);
               op->src_def[3] = old;
            }
            for (unsigned k = 0; k < 4; k++) {
               int d = op->src_def[k];
               if (d < 0)
                  continue;
               struct l2_def *def = &s->defs[d];
               if (def->num_readers >= 16) {
                  ok = false;
                  break;
               }
               def->readers[def->num_readers++] = oi;
            }
            if (!ok)
               break;
            if (a->dst.index != 31) {
               unsigned width = a->dst.write_low_sub_reg &&
                                   a->dst.write_high_sub_reg ? 2 : 1;
               int d = new_def(s, width, oi);
               if (d < 0) {
                  ok = false;
                  break;
               }
               if (pinned) {
                  s->defs[d].fixed_reg = a->dst.index;
                  s->defs[d].fixed_half = a->dst.write_low_sub_reg ? 0 : 1;
               }
               op->dst_def = d;
               pending[npend].op = oi;
               pending[npend].reg = a->dst.index;
               pending[npend].half_lo = a->dst.write_low_sub_reg;
               pending[npend].half_hi = a->dst.write_high_sub_reg;
               pending[npend].def = d;
               npend++;
            }
         }
         /* the slots of a packet read before any of them writes */
         for (unsigned n = 0; n < npend && ok; n++) {
            int sl = cur_slot(pending[n].reg);
            if (sl < 0) {
               ok = false;
               break;
            }
            if (s->defs[pending[n].def].fixed_reg >= 0) {
               /* a pinned register is written again: the write must come
                * after the write and every read of the value it replaces */
               int prev = cur.def[sl * 2 + (pending[n].half_lo ? 0 : 1)];
               struct l2_op *w = &s->ops[pending[n].op];
               if (prev >= 0) {
                  const struct l2_def *pd = &s->defs[prev];
                  if (pd->writer != pending[n].op)
                     w->pre[w->num_pre++] = pd->writer;
                  for (unsigned r = 0; r < pd->num_readers; r++)
                     if (pd->readers[r] != pending[n].op && w->num_pre < 24)
                        w->pre[w->num_pre++] = pd->readers[r];
                  if (w->num_pre >= 24) {
                     ok = false;
                     break;
                  }
               }
            }
            if (pending[n].half_lo)
               cur.def[sl * 2] = pending[n].def;
            if (pending[n].half_hi)
               cur.def[sl * 2 + 1] = pending[n].def;
         }
      }
      if (!ok)
         break;
      if (instr->dw.enable && !instr->dw.spill) {
         if (have_dw) {
            ok = false;
            break;
         }
         have_dw = true;
         struct l2_op *op = NEWOP(L2_DW);
         op->dw = instr->dw;
         int oi = s->num_ops++;
         unsigned regs = instr->dw.src_regs == FP_DW_REGS_R2_R3 ? 2 : 0;
         for (unsigned r = 0; r < 2 && ok; r++)
            for (unsigned h = 0; h < 2 && ok; h++) {
               int d = cur.def[(regs + r) * 2 + h];
               if (d < 0) {
                  ok = false;
                  break;
               }
               bool dup = false;
               for (unsigned n = 0; n < op->num_in; n++)
                  dup |= op->in[n] == d;
               int half = s->defs[d].width == 2 ? -1 : (int)h;
               if (!tie(s, d, oi, regs + r, half)) {
                  ok = false;
                  break;
               }
               if (!dup)
                  op->in[op->num_in++] = d;
            }
         if (!ok)
            break;
      }
   }
#undef NEWOP
   free(pk);
   free(mf);
   if (ok && !have_dw)
      ok = false;
   if (ok && !getenv("GRATE_LAYOUT_NOCSE"))
      l2_cse_interp(s);
   if (ok && !getenv("GRATE_LAYOUT_NOFOLD"))
      s->folded = l2_fold_copies(s);
   return ok;
}

/* The same varying component interpolated twice is the same value: later
 * readers read the first one, as NVIDIA's compiler does (one interpolation of
 * all four channels into R0-R3, then plain register reads). A variable
 * interpolated by an operation that then has nothing left to do goes away. */
static unsigned
l2_cse_interp(struct l2 *s)
{
   int first[256];
   unsigned removed = 0;

   memset(first, 0xff, sizeof(first));
   for (unsigned d = 0; d < s->num_defs; d++) {
      struct l2_def *y = &s->defs[d];
      if (y->interp_key < 0 || y->interp_key >= 256 || y->dead)
         continue;
      if (first[y->interp_key] < 0) {
         first[y->interp_key] = d;
         continue;
      }
      int xd = first[y->interp_key];
      bool used = false;
      for (unsigned i = 0; i < s->num_ops; i++) {
         struct l2_op *op = &s->ops[i];
         if (op->dead)
            continue;
         if (op->kind == L2_ALU) {
            for (unsigned k = 0; k < L2_SRCS; k++)
               if (op->src_def[k] == (int)d) {
                  op->src_def[k] = xd;
                  used = true;
               }
         } else {
            for (unsigned k = 0; k < op->num_in; k++)
               if (op->in[k] == (int)d) {
                  op->in[k] = xd;
                  used = true;
                  /* the unit now reads the first value: it sits on that
                   * value's row */
                  tie(s, xd, i, y->tie_k, y->tie_half);
               }
         }
      }
      (void)used;
      removed++;
   }
   /* an interpolation none of whose variables is the first one is not needed */
   for (unsigned i = 0; i < s->num_ops; i++) {
      struct l2_op *op = &s->ops[i];
      if (op->kind != L2_MFU || op->num_out == 0 || op->dead)
         continue;
      bool needed = false;
      for (unsigned k = 0; k < op->num_out; k++) {
         const struct l2_def *d = &s->defs[op->out[k]];
         needed |= first[d->interp_key >= 0 && d->interp_key < 256 ? d->interp_key : 0] == op->out[k];
      }
      /* an SFU result is not an interpolation: keep operations with none */
      bool all_interp = true;
      for (unsigned k = 0; k < op->num_out; k++)
         all_interp &= s->defs[op->out[k]].interp_key >= 0;
      if (all_interp && !needed) {
         op->dead = true;
         op->placed = true;
         removed++;
      }
   }
   return removed;
}

/* ---- folding copies ---- */

static bool
alu_is_plain_copy(const struct fp_alu_instr *a)
{
   const struct fp_alu_src_operand *s = a->src;
   return a->op == FP_ALU_OP_MAD && a->scale == FP_SCALE_NONE &&
          !a->dst.saturate && a->condition == FP_CONDITION_ALWAYS &&
          !s[0].negate && !s[0].absolute_value && !s[0].minus_one &&
          !s[0].scale_by_two && s[1].index == 31 && s[1].sub_reg_select_high &&
          !s[1].negate && !s[1].absolute_value && !s[1].minus_one &&
          !s[1].scale_by_two && s[2].index == 31 && !s[2].sub_reg_select_high &&
          !s[2].negate && !s[2].absolute_value && !s[2].minus_one &&
          !s[2].scale_by_two;
}

/* A copy whose source value has no other use is not needed: the slot that
 * computes the source writes the copy's destination instead. The conversion
 * between FP20 and an FX10 half happens in the write either way. Returns the
 * number of copies removed. */
static unsigned
l2_fold_copies(struct l2 *s)
{
   unsigned removed = 0;
   unsigned uses[L2_MAX_DEFS] = {0};

   for (unsigned i = 0; i < s->num_ops; i++) {
      const struct l2_op *op = &s->ops[i];
      unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
      for (unsigned k = 0; k < n; k++) {
         int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
         if (d >= 0)
            uses[d]++;
      }
   }
   for (unsigned c = 0; c < s->num_ops; c++) {
      struct l2_op *cp = &s->ops[c];
      if (cp->kind != L2_ALU || cp->dead || cp->dst_def < 0 ||
          !alu_is_plain_copy(&cp->alu) || cp->src_def[0] < 0 ||
          cp->src_def[1] >= 0 || cp->src_def[2] >= 0 || cp->src_def[3] >= 0)
         continue;
      struct l2_def *x = &s->defs[cp->src_def[0]];
      struct l2_def *y = &s->defs[cp->dst_def];
      /* a copy between values of the same width: the readers of the copy
       * read its source instead (copy propagation) */
      if (uses[cp->src_def[0]] != 1 && s->prop &&
          !x->dead && x->tie_k < 0 && x->fixed_reg < 0 && y->tie_k < 0 &&
          y->fixed_reg < 0 && y->width == x->width &&
          (x->width == 2 ? cp->alu.src[0].datatype == FP_DATATYPE_FP20
                         : cp->alu.src[0].datatype == FP_DATATYPE_FIXED10)) {
         bool only_alu = true;
         for (unsigned i = 0; i < s->num_ops && only_alu; i++) {
            const struct l2_op *op = &s->ops[i];
            if (op->dead)
               continue;
            if (op->kind == L2_ALU) {
               if (op->src_def[3] == cp->dst_def)
                  only_alu = false;
            } else {
               for (unsigned k = 0; k < op->num_in; k++)
                  if (op->in[k] == cp->dst_def)
                     only_alu = false;
            }
         }
         if (only_alu) {
            const int xd = cp->src_def[0], yd = cp->dst_def;
            for (unsigned i = 0; i < s->num_ops; i++) {
               struct l2_op *op = &s->ops[i];
               if (op->kind != L2_ALU || op->dead)
                  continue;
               for (unsigned k = 0; k < 3; k++)
                  if (op->src_def[k] == yd) {
                     op->src_def[k] = xd;
                     uses[xd]++;
                  }
            }
            uses[xd]--;
            y->dead = true;
            cp->dead = true;
            cp->placed = true;
            cp->dst_def = -1;
            cp->src_def[0] = -1;
            removed++;
            continue;
         }
      }
      if (uses[cp->src_def[0]] != 1 || x->tie_k >= 0 || x->fixed_reg >= 0 ||
          y->fixed_reg >= 0 || y->width > x->width || x->dead)
         continue;
      struct l2_op *p = &s->ops[x->writer];
      if (p->kind != L2_ALU || p->dead || p->dst_def != cp->src_def[0] ||
          p->alu.condition != FP_CONDITION_ALWAYS)
         continue;
      /* the copy must read the value in its own format */
      const struct fp_alu_src_operand *src = &cp->alu.src[0];
      if (x->width == 2 ? src->datatype != FP_DATATYPE_FP20
                        : src->datatype != FP_DATATYPE_FIXED10)
         continue;
      /* the producer must come first in the program, so that no other
       * ordering (pinned registers) is disturbed */
      if (x->writer > (int)c)
         continue;
      p->dst_def = cp->dst_def;
      y->writer = x->writer;
      x->dead = true;
      cp->dead = true;
      cp->placed = true;
      cp->dst_def = -1;
      cp->src_def[0] = -1;
      removed++;
   }
   return removed;
}

/* ---- scheduling ---- */

static bool
visible(const struct l2 *s, int def, int64_t g, unsigned step, bool unit_reader)
{
   const struct l2_def *d = &s->defs[def];
   if (!d->written)
      return false;
   bool earlier_round = (unsigned)(d->step / s->rows) < step / s->rows;
   if (unit_reader) {
      /* a unit reads at the start of its round: only earlier rounds, or
       * another unit's write on the same step (experiments d1-d4) */
      return earlier_round ||
             (s->ops[d->writer].kind != L2_ALU && d->step == (int)step);
   }
   if (s->ops[d->writer].kind != L2_ALU)
      return true;
   return earlier_round || g >= d->issue + 2;
}

/* Source k of an ALU operation can come from the packet right before it: the
 * producer sits in the previous packet of the same round and the value has the
 * format that packet's slot wrote (docs/grate-alu-latency.md). Returns the
 * slot, or -1. */
static int
fwd_slot(const struct l2 *s, const struct l2_op *op, unsigned k, int64_t g,
         unsigned step)
{
   if (op->kind != L2_ALU || k >= 3 || op->src_def[k] < 0)
      return -1;
   const struct l2_def *d = &s->defs[op->src_def[k]];
   if (getenv("GRATE_LAYOUT_NOFWD"))
      return -1;
   if (getenv("GRATE_LAYOUT_FWD_SAME_STEP") &&
       s->defs[op->src_def[k]].step != (int)step)
      return -1;
   if (!d->written || s->ops[d->writer].kind != L2_ALU)
      return -1;
   const struct l2_op *p = &s->ops[d->writer];
   if (!p->placed || d->issue != g - 1 ||
       (unsigned)(d->step / s->rows) != step / s->rows)
      return -1;
   const struct fp_alu_src_operand *src = &op->alu.src[k];
   if (d->width == 2 ? src->datatype != FP_DATATYPE_FP20
                     : src->datatype != FP_DATATYPE_FIXED10)
      return -1;
   return p->slot;
}

static bool
ops_ready(const struct l2 *s, const struct l2_op *op, int64_t g, unsigned step)
{
   for (unsigned i = 0; i < op->num_pre; i++)
      if (!s->ops[op->pre[i]].placed)
         return false;
   unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
   for (unsigned k = 0; k < n; k++) {
      int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
      if (d >= 0 && !visible(s, d, g, step, op->kind != L2_ALU) &&
          fwd_slot(s, op, k, g, step) < 0)
         return false;
   }
   return true;
}

static void
place_reads(struct l2 *s, struct l2_op *op, int64_t g, int op_index)
{
   unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
   for (unsigned k = 0; k < n; k++) {
      int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
      if (d < 0)
         continue;
      struct l2_def *def = &s->defs[d];
      def->uses_left--;
      if (g >= def->last_read) {
         def->last_read = g;
         def->last_reader = op_index;
      }
   }
}

static bool
unit_row_ok(struct l2 *s, const struct l2_op *op, unsigned step)
{
   int row = s->cls_row[cls_find(s, op->cls)];
   if (getenv("GRATE_LAYOUT_ROW0") && step % s->rows)
      return false;
   return row < 0 || (unsigned)row == step % s->rows;
}

/* Would the tied values of class cls collide with a live or reserved value of
 * another class if the class used this row? */
static bool
class_row_conflict(struct l2 *s, int cls, unsigned row)
{
   for (unsigned a = 0; a < s->num_defs; a++) {
      const struct l2_def *d = &s->defs[a];
      if (d->tie_k < 0 || cls_find(s, d->cls) != cls || s->ops[d->writer].dead)
         continue;
      for (unsigned b = 0; b < s->num_defs; b++) {
         const struct l2_def *e = &s->defs[b];
         if (b == a || e->tie_k != d->tie_k || s->ops[e->writer].dead)
            continue;
         int ecls = cls_find(s, e->cls);
         if (ecls == cls || s->cls_row[ecls] != (int)row)
            continue;
         bool halves = d->tie_half < 0 || e->tie_half < 0 ||
                       d->tie_half == e->tie_half;
         if (!halves)
            continue;
         /* live, or promised to its row and not written yet */
         if (!e->written || e->uses_left > 0)
            return true;
      }
   }
   return false;
}

/* A unit operation writes its row registers at the start of its round, before
 * any ALU packet of the round: the previous value of such a register must have
 * no read at or after that moment (rs is the position of the first packet of
 * the round), and no other value tied to it may be waiting. */
static bool
unit_loc_conflict(struct l2 *s, const struct l2_op *op, unsigned row,
                  int64_t rs)
{
   int cls = cls_find(s, op->cls);

   for (unsigned a = 0; a < s->num_defs; a++) {
      const struct l2_def *d = &s->defs[a];
      if (d->tie_k < 0 || cls_find(s, d->cls) != cls || s->ops[d->writer].dead)
         continue;
      /* only what this unit writes */
      bool writes = false;
      for (unsigned k = 0; k < op->num_out; k++)
         writes |= op->out[k] == (int)a;
      if (!writes)
         continue;
      for (unsigned b = 0; b < s->num_defs; b++) {
         const struct l2_def *e = &s->defs[b];
         if (b == a || e->tie_k != d->tie_k || s->ops[e->writer].dead)
            continue;
         int ecls = cls_find(s, e->cls);
         if (ecls == cls || s->cls_row[ecls] != (int)row)
            continue;
         if (!(d->tie_half < 0 || e->tie_half < 0 || d->tie_half == e->tie_half))
            continue;
         if (!e->written || e->uses_left > 0 || e->last_read >= rs)
            return true;
      }
   }
   return false;
}

/* Place a unit operation (MFU or fetch) on step. */
static bool
place_unit(struct l2 *s, struct l2_op *op, unsigned step, int64_t gfirst)
{
   if (!unit_row_ok(s, op, step) || !ops_ready(s, op, gfirst, step))
      return false;
   int cls = cls_find(s, op->cls);
   if (s->cls_row[cls] < 0) {
      if (class_row_conflict(s, cls, step % s->rows))
         return false;
      s->cls_row[cls] = step % s->rows;
   }
   if (unit_loc_conflict(s, op, step % s->rows, gfirst))
      return false;
   place_reads(s, op, gfirst, op->orig);
   for (unsigned k = 0; k < op->num_out; k++) {
      struct l2_def *d = &s->defs[op->out[k]];
      d->written = true;
      d->step = step;
      d->issue = gfirst;
      d->land = gfirst;
      d->last_read = gfirst;
   }
   op->placed = true;
   op->step = step;
   return true;
}

static bool
place_alu(struct l2 *s, struct l2_op *op, unsigned step, int64_t g,
          unsigned packet, unsigned slot)
{
   if (!ops_ready(s, op, g, step))
      return false;
   if (op->dst_def >= 0 && s->defs[op->dst_def].tie_k >= 0) {
      int cls = cls_find(s, s->defs[op->dst_def].cls);
      if (s->cls_row[cls] < 0) {
         /* the unit that uses it goes on the next step or one after it */
         int row = -1;
         for (unsigned j = 0; j < s->rows && row < 0; j++)
            if (!class_row_conflict(s, cls, (step + 1 + j) % s->rows) &&
                (!getenv("GRATE_LAYOUT_ROW0") || (step + 1 + j) % s->rows == 0))
               row = (step + 1 + j) % s->rows;
         if (row < 0)
            return false;
         s->cls_row[cls] = row;
      }
   }
   for (unsigned k = 0; k < 3; k++) {
      op->fwd[k] = -1;
      if (op->src_def[k] >= 0 &&
          !visible(s, op->src_def[k], g, step, false))
         op->fwd[k] = fwd_slot(s, op, k, g, step);
   }
   place_reads(s, op, g, op->orig);
   if (op->dst_def >= 0) {
      struct l2_def *d = &s->defs[op->dst_def];
      d->written = true;
      d->step = step;
      d->issue = g;
      d->land = g + 2;
      d->last_read = g + 2;
   }
   op->placed = true;
   op->step = step;
   op->packet = packet;
   op->slot = slot;
   return true;
}

static void
compute_heights(struct l2 *s)
{
   for (int i = s->num_ops - 1; i >= 0; i--) {
      struct l2_op *op = &s->ops[i];
      int h = 1;
      for (unsigned j = i + 1; j < s->num_ops; j++) {
         const struct l2_op *c = &s->ops[j];
         unsigned n = c->kind == L2_ALU ? L2_SRCS : c->num_in;
         for (unsigned k = 0; k < n; k++) {
            int d = c->kind == L2_ALU ? c->src_def[k] : c->in[k];
            if (d >= 0 && s->defs[d].writer == i)
               h = MAX2(h, c->height + 1);
         }
      }
      op->height = h;
   }
}

static unsigned
first_waiting(const struct l2 *s)
{
   for (unsigned i = 0; i < s->num_ops; i++)
      if (!s->ops[i].placed && s->ops[i].kind != L2_DW)
         return i;
   return s->num_ops;
}

static bool
l2_schedule(struct l2 *s)
{
   /* DW tie: the output is read from row 0 */
   for (unsigned i = 0; i < s->num_ops; i++)
      if (s->ops[i].kind == L2_DW)
         s->cls_row[cls_find(s, s->ops[i].cls)] = 0;
   for (unsigned d = 0; d < s->num_defs; d++) {
      s->defs[d].uses_left = 0;
      s->defs[d].written = false;
      s->defs[d].reg = s->defs[d].half = -1;
   }
   for (unsigned i = 0; i < s->num_ops; i++) {
      struct l2_op *op = &s->ops[i];
      op->placed = op->dead;
      op->fwd[0] = op->fwd[1] = op->fwd[2] = -1;
      unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
      for (unsigned k = 0; k < n; k++) {
         int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
         if (d >= 0)
            s->defs[d].uses_left++;
      }
   }
   compute_heights(s);
   s->total_packets = 0;
   s->steps = 0;
   memset(s->step_packets, 0, sizeof(s->step_packets));
   for (unsigned i = 0; i < L2_MAX_STEPS; i++)
      s->step_mfu[i] = s->step_tex[i] = -1;
   memset(s->packet_op, 0xff, sizeof(s->packet_op));

   unsigned remaining = 0;
   for (unsigned i = 0; i < s->num_ops; i++)
      remaining += s->ops[i].kind != L2_DW && !s->ops[i].dead;
   unsigned idle = 0;
   unsigned last_busy_step = 0;

   for (unsigned t = 0; remaining && t < L2_MAX_STEPS - s->rows; t++) {
      int64_t gfirst = s->total_packets;
      s->step_first[t] = gfirst;
      bool progress = false;

      /* unit operations first, the best of each kind */
      for (int kind = L2_MFU; kind <= L2_TEX; kind++) {
         int best = -1;
         for (unsigned i = 0; i < s->num_ops; i++) {
            struct l2_op *op = &s->ops[i];
            if (op->placed || (int)op->kind != kind)
               continue;
            if (i >= first_waiting(s) + s->window)
               continue;
            if (!unit_row_ok(s, op, t) || !ops_ready(s, op, gfirst, t))
               continue;
            if (best < 0 || op->height > s->ops[best].height)
               best = i;
         }
         if (best >= 0 &&
             place_unit(s, &s->ops[best], t, s->step_first[t - t % s->rows])) {
            if (kind == L2_MFU)
               s->step_mfu[t] = best;
            else
               s->step_tex[t] = best;
            remaining--;
            progress = true;
         }
      }

      /* ALU packets */
      for (unsigned p = 0; p < 3; p++) {
         int64_t g = s->total_packets;
         int picked[4];
         unsigned np = 0;
         for (unsigned slot = 0; slot < 4; slot++) {
            int best = -1;
            for (unsigned i = 0; i < s->num_ops; i++) {
               struct l2_op *op = &s->ops[i];
               if (op->placed || op->kind != L2_ALU)
                  continue;
               if (i >= first_waiting(s) + s->window)
                  continue;
               if (!ops_ready(s, op, g, t))
                  continue;
               if (best < 0 || op->height > s->ops[best].height)
                  best = i;
            }
            /* try the best candidates in order until one is placed */
            bool placed = false;
            while (best >= 0) {
               if (place_alu(s, &s->ops[best], t, g, p, np)) {
                  picked[np++] = best;
                  placed = true;
                  break;
               }
               /* no free location for it now: look at the next candidate */
               int next = -1;
               for (unsigned i = 0; i < s->num_ops; i++) {
                  struct l2_op *op = &s->ops[i];
                  if (op->placed || op->kind != L2_ALU || (int)i == best)
                     continue;
                  if (i >= first_waiting(s) + s->window)
                     continue;
                  if (!ops_ready(s, op, g, t))
                     continue;
                  if (s->ops[best].height < op->height ||
                      (s->ops[best].height == op->height && (int)i < best))
                     continue;
                  if (next < 0 || op->height > s->ops[next].height)
                     next = i;
               }
               best = next;
            }
            if (!placed)
               break;
         }
         if (np) {
            for (unsigned k = 0; k < np; k++) {
               s->packet_op[t][p][k] = picked[k];
               s->ops[picked[k]].slot = k;
               remaining--;
            }
            s->step_packets[t]++;
            s->total_packets++;
            progress = true;
            continue;
         }
         /* nothing could go: a bubble if an operation only waits for the
          * latency of an earlier packet of this round */
         bool waiting = false;
         if (p < 2 && (t + 1) % s->rows != 0 && s->step_packets[t] > 0)
            for (unsigned i = 0; i < s->num_ops && !waiting; i++) {
               struct l2_op *op = &s->ops[i];
               if (op->placed || op->kind == L2_DW)
                  continue;
               if (ops_ready(s, op, g + 1, t) && !ops_ready(s, op, g, t))
                  waiting = true;
            }
         if (waiting) {
            s->step_packets[t]++;
            s->total_packets++;
            continue;
         }
         break;
      }

      /* a unit operation that waits for the packet before the next step */
      if (s->step_packets[t] && s->step_packets[t] < 3 && (t + 1) % s->rows != 0) {
         bool waiting = false;
         for (unsigned i = 0; i < s->num_ops && !waiting; i++) {
            struct l2_op *op = &s->ops[i];
            if (op->placed || (op->kind != L2_MFU && op->kind != L2_TEX))
               continue;
            if (unit_row_ok(s, op, t + 1) &&
                !ops_ready(s, op, s->total_packets, t + 1) &&
                ops_ready(s, op, s->total_packets + 1, t + 1))
               waiting = true;
         }
         if (waiting) {
            s->step_packets[t]++;
            s->total_packets++;
         }
      }

      s->steps = t + 1;
      if (progress) {
         idle = 0;
         last_busy_step = t;
      } else if (++idle > 2 * s->rows + 4) {
         if (s->verbose) {
            fprintf(stderr, "GRATE LAYOUT2: rows %u stuck at step %u, %u ops left:\n",
                    s->rows, t, remaining);
            for (unsigned i = 0; i < s->num_ops; i++) {
               const struct l2_op *op = &s->ops[i];
               if (op->placed || op->kind == L2_DW)
                  continue;
               unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
               fprintf(stderr, "   op %u kind %d cls_row %d:", i, op->kind,
                       s->cls_row[cls_find(s, op->cls)]);
               for (unsigned k = 0; k < n; k++) {
                  int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
                  if (d >= 0)
                     fprintf(stderr, " in def%d(%s)", d,
                             s->defs[d].written ? "written" : "NOT written");
               }
               if (op->kind == L2_ALU && op->dst_def >= 0) {
                  const struct l2_def *d = &s->defs[op->dst_def];
                  fprintf(stderr, " out def%d tie k%d h%d w%u", op->dst_def,
                          d->tie_k, d->tie_half, d->width);
               }
               fprintf(stderr, "\n");
            }
         }
         return false;
      }
      (void)last_busy_step;
   }
   if (remaining)
      return false;

   /* the output write: every other operation is placed; it goes on the first
    * step of the round of the last step that does something */
   unsigned last = 0;
   for (unsigned t = 0; t < s->steps; t++)
      if (s->step_packets[t] || s->step_mfu[t] >= 0 || s->step_tex[t] >= 0)
         last = t;
   unsigned dw_step = last - last % s->rows;
   for (unsigned i = 0; i < s->num_ops; i++) {
      struct l2_op *op = &s->ops[i];
      if (op->kind != L2_DW)
         continue;
      for (unsigned k = 0; k < op->num_in; k++)
         if (!s->defs[op->in[k]].written)
            return false;
      op->step = dw_step;
      op->placed = true;
   }
   s->steps = dw_step + s->rows;
   if (s->steps > L2_MAX_STEPS)
      return false;
   /* every round starts with an ALU packet */
   for (unsigned t = 0; t < s->steps; t += s->rows)
      if (!s->step_packets[t]) {
         s->step_packets[t] = 1;
         s->total_packets++;
      }
   /* recompute the first packet positions */
   int64_t g = 0;
   for (unsigned t = 0; t < s->steps; t++) {
      s->step_first[t] = g;
      g += s->step_packets[t];
   }
   return true;
}

/* ---- register allocation on the finished schedule ---- */

struct l2_occ {
   int64_t land, end;
   int def;
};

static bool
occ_fits(const struct l2 *s, const struct l2_occ *occ, unsigned n, int def,
         int64_t land, int64_t end)
{
   const struct l2_def *nd = &s->defs[def];
   for (unsigned i = 0; i < n; i++) {
      if (land > occ[i].end || end < occ[i].land)
         continue;
      /* an SFU reads and writes the same register at the same moment */
      const struct l2_def *od = &s->defs[occ[i].def];
      if (land == occ[i].end && od->last_reader == nd->writer &&
          s->ops[nd->writer].kind != L2_ALU && end >= land)
         continue;
      return false;
   }
   return true;
}

static bool
l2_allocate(struct l2 *s)
{
   static struct l2_occ occ[48][128];
   unsigned nocc[48] = {0};
   unsigned regs = s->rows * 4;

   /* a value that is only read through forwards needs no register */
   {
      unsigned uses[L2_MAX_DEFS] = {0}, fwds[L2_MAX_DEFS] = {0};
      for (unsigned i = 0; i < s->num_ops; i++) {
         const struct l2_op *op = &s->ops[i];
         unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
         for (unsigned k = 0; k < n; k++) {
            int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
            if (d < 0)
               continue;
            uses[d]++;
            if (op->kind == L2_ALU && k < 3 && op->fwd[k] >= 0)
               fwds[d]++;
         }
      }
      for (unsigned d = 0; d < s->num_defs; d++) {
         struct l2_def *def = &s->defs[d];
         def->reg_free = !def->dead && def->width == 2 && def->tie_k < 0 &&
                         def->fixed_reg < 0 && uses[d] > 0 &&
                         uses[d] == fwds[d] &&
                         s->ops[def->writer].kind == L2_ALU &&
                         s->ops[def->writer].alu.condition == FP_CONDITION_ALWAYS;
      }
   }

   /* final positions of everything */
   int64_t total = 0;
   for (unsigned t = 0; t < s->steps; t++)
      total += s->step_packets[t];
   for (unsigned d = 0; d < s->num_defs; d++) {
      struct l2_def *def = &s->defs[d];
      const struct l2_op *w = &s->ops[def->writer];
      /* a unit acts at the start of its round */
      def->issue = w->kind == L2_ALU
                      ? s->step_first[w->step] + w->packet
                      : s->step_first[w->step - w->step % s->rows];
      /* a round boundary completes the pending writes of the round before it
       * (experiment dxa), so a result lands at the end of its round at the
       * latest */
      int64_t round_end = (unsigned)((w->step / s->rows + 1) * s->rows) < s->steps
                             ? s->step_first[(w->step / s->rows + 1) * s->rows]
                             : total;
      def->land = w->kind == L2_ALU ? MIN2(def->issue + 2, round_end)
                                    : def->issue;
      /* a round boundary lets a later round see it before it lands */
      def->start = w->kind == L2_ALU ? def->issue + 1 : def->issue;
      def->last_read = def->land;
      def->last_reader = -1;
   }
   for (unsigned i = 0; i < s->num_ops; i++) {
      const struct l2_op *op = &s->ops[i];
      int64_t g = op->kind == L2_ALU
                     ? s->step_first[op->step] + op->packet
                     : s->step_first[op->step - op->step % s->rows];
      unsigned n = op->kind == L2_ALU ? L2_SRCS : op->num_in;
      for (unsigned k = 0; k < n; k++) {
         int d = op->kind == L2_ALU ? op->src_def[k] : op->in[k];
         if (d < 0)
            continue;
         struct l2_def *def = &s->defs[d];
         int64_t at = op->kind == L2_DW ? total + 2 : g;
         if (at >= def->last_read) {
            def->last_read = at;
            def->last_reader = i;
         }
      }
   }

   /* tied and pinned values first, they have no choice */
   for (unsigned d = 0; d < s->num_defs; d++) {
      struct l2_def *def = &s->defs[d];
      if ((def->tie_k < 0 && def->fixed_reg < 0) || s->ops[def->writer].dead)
         continue;
      unsigned reg, half;
      if (def->fixed_reg >= 0) {
         reg = def->fixed_reg;
         half = def->fixed_half;
         if (!def->written)
            return false;
      } else {
         int row = s->cls_row[cls_find(s, def->cls)];
         if (row < 0 || !def->written) {
            if (s->verbose)
               fprintf(stderr, "GRATE LAYOUT2: tied value def%d (writer op %d kind %d) has row %d written %d\n",
                       d, def->writer, s->ops[def->writer].kind, row, def->written);
            return false;
         }
         reg = row * 4 + def->tie_k;
         half = def->tie_half < 0 ? 0 : def->tie_half;
      }
      for (unsigned h = 0; h < 2; h++) {
         if (def->width == 1 && h != half)
            continue;
         unsigned loc = reg < 16 ? reg * 2 + h : 32 + (reg - 64) * 2 + h;
         if (!occ_fits(s, occ[loc], nocc[loc], d, def->start, def->last_read)) {
            if (s->verbose) {
               fprintf(stderr, "GRATE LAYOUT2: tied value def%d (writer op %d kind %d, interp key %d, [%lld..%lld]) cannot have r%u; there:",
                       d, def->writer, s->ops[def->writer].kind, def->interp_key,
                       (long long)def->start, (long long)def->last_read, reg);
               for (unsigned i = 0; i < nocc[loc]; i++)
                  fprintf(stderr, " def%d(writer op %d kind %d key %d [%lld..%lld])", occ[loc][i].def,
                          s->defs[occ[loc][i].def].writer, s->ops[s->defs[occ[loc][i].def].writer].kind,
                          s->defs[occ[loc][i].def].interp_key, (long long)occ[loc][i].land, (long long)occ[loc][i].end);
               fprintf(stderr, "\n");
            }
            return false;
         }
         occ[loc][nocc[loc]++] = (struct l2_occ){def->start, def->last_read, (int)d};
      }
      def->reg = reg;
      def->half = def->width == 2 ? 0 : half;
   }
   /* the others, in the order they are written */
   bool done[L2_MAX_DEFS] = {false};
   for (unsigned n = 0; n < s->num_defs; n++) {
      int pick = -1;
      for (unsigned d = 0; d < s->num_defs; d++) {
         if (done[d] || s->defs[d].tie_k >= 0 || s->defs[d].fixed_reg >= 0 ||
             s->defs[d].reg_free || s->defs[d].dead)
            continue;
         if (pick < 0 || s->defs[d].land < s->defs[pick].land)
            pick = d;
      }
      if (pick < 0)
         break;
      done[pick] = true;
      struct l2_def *def = &s->defs[pick];
      bool placed = false;
      /* the register that was free the longest */
      int best_loc = -1;
      int64_t best_gap = -1;
      for (unsigned r = 0; r < regs && !placed; r++) {
         for (unsigned h = 0; h < (def->width == 2 ? 1u : 2u); h++) {
            bool fits = true;
            int64_t last_end = -1;
            for (unsigned hh = 0; hh < 2; hh++) {
               if (def->width == 1 && hh != h)
                  continue;
               unsigned loc = r * 2 + hh;
               if (!occ_fits(s, occ[loc], nocc[loc], pick, def->start,
                             def->last_read)) {
                  fits = false;
                  break;
               }
               for (unsigned i = 0; i < nocc[loc]; i++)
                  if (occ[loc][i].end < def->start)
                     last_end = MAX2(last_end, occ[loc][i].end);
            }
            if (!fits)
               continue;
            int64_t gap = def->start - last_end;
            if (best_loc < 0 || gap < best_gap) {   /* best fit */
               best_loc = r * 2 + h;
               best_gap = gap;
            }
         }
      }
      if (best_loc < 0) {
         if (s->verbose)
            fprintf(stderr, "GRATE LAYOUT2: no register for def%d (%lld..%lld) with %u\n",
                    pick, (long long)def->start, (long long)def->last_read, regs);
         return false;
      }
      def->reg = best_loc / 2;
      def->half = def->width == 2 ? 0 : best_loc % 2;
      for (unsigned hh = 0; hh < 2; hh++) {
         if (def->width == 1 && hh != (unsigned)def->half)
            continue;
         unsigned loc = def->reg * 2 + hh;
         if (nocc[loc] >= 128)
            return false;
         occ[loc][nocc[loc]++] = (struct l2_occ){def->start, def->last_read, pick};
      }
   }
   return true;
}

/* ---- rebuilding the program ---- */

static void
rewrite_src(const struct l2 *s, struct fp_alu_src_operand *src, int def)
{
   if (def < 0)
      return;
   const struct l2_def *d = &s->defs[def];
   src->index = d->reg;
   if (d->width == 1) {
      src->datatype = FP_DATATYPE_FIXED10;
      src->sub_reg_select_high = d->half;
   }
}

static bool
l2_rebuild(struct l2 *s, struct grate_fp_shader *fp, struct list_head *old_steps,
           struct list_head *old_alus, struct list_head *old_mfus)
{
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);
   unsigned address = 0;
   for (unsigned t = 0; t < s->steps; t++) {
      struct fp_instr *instr = CALLOC_STRUCT(fp_instr);
      if (!instr)
         return false;
      list_inithead(&instr->link);
      if (s->step_mfu[t] >= 0) {
         struct fp_mfu_instr *copy = CALLOC_STRUCT(fp_mfu_instr);
         if (!copy)
            return false;
         *copy = *s->ops[s->step_mfu[t]].mfu;
         list_inithead(&copy->link);
         instr->mfu_sched.address = list_length(&fp->mfu_instructions);
         instr->mfu_sched.num_instructions = 1;
         list_addtail(&copy->link, &fp->mfu_instructions);
      }
      if (s->step_tex[t] >= 0)
         instr->tex = s->ops[s->step_tex[t]].tex;
      for (unsigned i = 0; i < s->num_ops; i++)
         if (s->ops[i].kind == L2_DW && s->ops[i].step == (int)t)
            instr->dw = s->ops[i].dw;
      instr->alu_sched.address = address;
      instr->alu_sched.num_instructions = s->step_packets[t];
      for (unsigned p = 0; p < s->step_packets[t]; p++) {
         struct fp_alu_instr_packet *packet = CALLOC_STRUCT(fp_alu_instr_packet);
         if (!packet)
            return false;
         list_inithead(&packet->link);
         for (unsigned sl = 0; sl < 4; sl++)
            for (unsigned k = 0; k < 4; k++)
               packet->slots[sl].src[k].index = 31;
         for (unsigned sl = 0; sl < 4; sl++) {
            int oi = p < 3 ? s->packet_op[t][p][sl] : -1;
            if (oi < 0)
               continue;
            const struct l2_op *op = &s->ops[oi];
            struct fp_alu_instr *a = &packet->slots[sl];
            *a = op->alu;
            for (unsigned k = 0; k < 3; k++) {
               rewrite_src(s, &a->src[k], op->src_def[k]);
               if (op->fwd[k] >= 0) {
                  const struct l2_def *d = &s->defs[op->src_def[k]];
                  a->src[k].index = 24 + op->fwd[k];
                  a->src[k].datatype = d->width == 2 ? FP_DATATYPE_FP20
                                                     : FP_DATATYPE_FIXED10;
                  a->src[k].sub_reg_select_high = d->width == 2 ? 0 : d->half;
               }
            }
            if (op->dst_def >= 0) {
               const struct l2_def *d = &s->defs[op->dst_def];
               a->dst.index = d->reg_free ? 31 : d->reg;
               a->dst.write_low_sub_reg = d->width == 2 || d->half == 0;
               a->dst.write_high_sub_reg = d->width == 2 || d->half == 1;
            }
         }
         list_addtail(&packet->link, &fp->alu_instructions);
         address++;
      }
      list_addtail(&instr->link, &fp->fp_instructions);
   }
   (void)old_steps;
   (void)old_alus;
   (void)old_mfus;
   return true;
}

static void
free_lists(struct list_head *steps, struct list_head *alus,
           struct list_head *mfus)
{
   list_for_each_entry_safe(struct fp_instr, i, steps, link) {
      list_del(&i->link);
      FREE(i);
   }
   list_for_each_entry_safe(struct fp_alu_instr_packet, p, alus, link) {
      list_del(&p->link);
      FREE(p);
   }
   list_for_each_entry_safe(struct fp_mfu_instr, m, mfus, link) {
      list_del(&m->link);
      FREE(m);
   }
}

void
grate_fp_ir_free(struct list_head *steps, struct list_head *alus,
                 struct list_head *mfus)
{
   free_lists(steps, alus, mfus);
}

bool
grate_fp_ir_clone(const struct grate_fp_shader *fp, struct list_head *steps,
                  struct list_head *alus, struct list_head *mfus)
{
   list_inithead(steps);
   list_inithead(alus);
   list_inithead(mfus);
   list_for_each_entry (struct fp_instr, i, &fp->fp_instructions, link) {
      struct fp_instr *c = CALLOC_STRUCT(fp_instr);
      if (!c)
         goto fail;
      *c = *i;
      list_inithead(&c->link);
      list_addtail(&c->link, steps);
   }
   list_for_each_entry (struct fp_alu_instr_packet, p, &fp->alu_instructions,
                        link) {
      struct fp_alu_instr_packet *c = CALLOC_STRUCT(fp_alu_instr_packet);
      if (!c)
         goto fail;
      *c = *p;
      list_inithead(&c->link);
      list_addtail(&c->link, alus);
   }
   list_for_each_entry (struct fp_mfu_instr, m, &fp->mfu_instructions, link) {
      struct fp_mfu_instr *c = CALLOC_STRUCT(fp_mfu_instr);
      if (!c)
         goto fail;
      *c = *m;
      list_inithead(&c->link);
      list_addtail(&c->link, mfus);
   }
   return true;
fail:
   free_lists(steps, alus, mfus);
   return false;
}

bool
grate_fp_layout_nvidia(struct grate_fp_shader *fp, uint64_t seq_signature,
                       bool verbose, unsigned old_steps, unsigned old_packets)
{
   static struct l2 storage;
   struct l2 *s = &storage;
   static const unsigned windows[] = {3, 5, 8, 12, 20, 40, 1000};
   struct cand {
      unsigned rows, window, steps, packets;
      bool prop;
   } cands[8 * ARRAY_SIZE(windows)];
   unsigned num_cands = 0;

   for (unsigned rows = 1; rows <= 4; rows++) {
      for (unsigned w = 0; w < 2 * ARRAY_SIZE(windows); w++) {
         memset(s, 0, sizeof(*s));
         s->rows = rows;
         s->window = windows[w % ARRAY_SIZE(windows)];
         s->prop = w >= ARRAY_SIZE(windows);
         s->verbose = verbose && getenv("GRATE_LAYOUT_VERBOSE");
         if (!l2_extract(s, fp)) {
            if (verbose)
               fprintf(stderr, "GRATE LAYOUT2: cannot extract the program\n");
            return false;
         }
         if (!l2_schedule(s)) {
            if (verbose)
               fprintf(stderr, "GRATE LAYOUT2: rows %u window %u: no schedule\n", rows, windows[w]);
            continue;
         }
         if (!l2_allocate(s)) {
            if (verbose)
               fprintf(stderr, "GRATE LAYOUT2: rows %u window %u: no registers\n",
                       rows, windows[w]);
            continue;
         }
         cands[num_cands++] = (struct cand){rows, windows[w % ARRAY_SIZE(windows)],
                                            s->steps,
                                            (unsigned)s->total_packets,
                                            s->prop};
      }
   }
   /* best first: fewest steps, then packets, then rows */
   for (unsigned a = 0; a < num_cands; a++)
      for (unsigned b = a + 1; b < num_cands; b++)
         if (cands[b].steps < cands[a].steps ||
             (cands[b].steps == cands[a].steps &&
              (cands[b].packets < cands[a].packets ||
               (cands[b].packets == cands[a].packets &&
                cands[b].rows < cands[a].rows)))) {
            struct cand t = cands[a];
            cands[a] = cands[b];
            cands[b] = t;
         }

   const char *only = getenv("GRATE_LAYOUT_ONLY");
   unsigned only_rows = 0, only_window = 0;
   if (only)
      sscanf(only, "%u,%u", &only_rows, &only_window);
   for (unsigned c = 0; c < num_cands; c++) {
      /* one epoch only: the epoch splitter has never seen these layouts */
      if (cands[c].packets > 64 || cands[c].steps > 64)
         continue;
      if (!only && (cands[c].steps > old_steps ||
                    (cands[c].steps == old_steps && cands[c].packets >= old_packets))) {
         if (verbose)
            fprintf(stderr, "GRATE LAYOUT2: no candidate beats the old layout (%u steps, %u packets)\n",
                    old_steps, old_packets);
         return false;
      }
      if (only && (cands[c].rows != only_rows ||
                   (only_window && cands[c].window != only_window)))
         continue;
      memset(s, 0, sizeof(*s));
      s->rows = cands[c].rows;
      s->window = cands[c].window;
      s->prop = cands[c].prop;
      if (!l2_extract(s, fp) || !l2_schedule(s) || !l2_allocate(s))
         continue;

      struct list_head old_steps, old_alus, old_mfus;
      list_replace(&fp->fp_instructions, &old_steps);
      list_replace(&fp->alu_instructions, &old_alus);
      list_replace(&fp->mfu_instructions, &old_mfus);
      unsigned old_rows = fp->info.num_rows;
      fp->info.num_rows = cands[c].rows;
      bool ok = l2_rebuild(s, fp, &old_steps, &old_alus, &old_mfus);
      uint64_t sig = 0;
      const char *why = NULL;
      if (ok)
         ok = grate_fp_layout_signature(fp, false, &sig, &why) &&
              sig == seq_signature;
      if (verbose)
         fprintf(stderr, "GRATE LAYOUT2: rows %u window %u%s, %u steps, %u packets: %s\n",
                 cands[c].rows, cands[c].window, cands[c].prop ? " +prop" : "", cands[c].steps,
                 cands[c].packets, ok ? "validated" : "REJECTED by the evaluator");
      if (ok) {
         free_lists(&old_steps, &old_alus, &old_mfus);
         return true;
      }
      free_lists(&fp->fp_instructions, &fp->alu_instructions,
                 &fp->mfu_instructions);
      list_replace(&old_steps, &fp->fp_instructions);
      list_replace(&old_alus, &fp->alu_instructions);
      list_replace(&old_mfus, &fp->mfu_instructions);
      fp->info.num_rows = old_rows;
   }
   return false;
}
