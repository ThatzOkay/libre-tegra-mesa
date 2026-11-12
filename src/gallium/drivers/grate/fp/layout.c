/*
 * Symbolic evaluation of fragment program layouts, and (layout2.c) a layout
 * that follows NVIDIA's: the unit operations (interpolation, SFU, fetch) sit
 * on any step and work on the register row `step % rows`.
 *
 * Rules (measured, docs/grate-alu-latency.md):
 *  - a unit operation on step s reads and writes R(4r)..R(4r+3) of the row
 *    r = s % rows, before the ALU packets of the step run;
 *  - an ALU result is visible to a read two packets later, or in a later
 *    round (a round is `rows` steps); a read one packet later returns the old
 *    value; a step change inside a round gives no extra latency;
 *  - the output write reads R2/R3 (or R0/R1) of its row at the end of its
 *    round and sees everything issued up to then.
 */

#include <stdio.h>
#include <stdlib.h>

#include "util/list.h"
#include "util/macros.h"
#include "util/u_math.h"

#include "fp/fpir.h"
#include "fp/layout.h"
#include "grate_compiler.h"

#define LO_MAX_WRITES 4096

struct lo_write {
   int loc;            /* register * 2 + half */
   uint64_t value;
   unsigned width;     /* 2: FP20 in both halves, 1: FX10 in one half */
   int64_t issue;      /* packet position of the issue */
   int64_t land;       /* position from which it is visible in its round */
   int round;
   unsigned seq;
};

struct lo_eval {
   bool sequential;
   unsigned rows;
   struct lo_write writes[LO_MAX_WRITES];
   unsigned num_writes;
   const char *why;
};

static uint64_t
mix(uint64_t a, uint64_t b)
{
   uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ull;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebull;
   x ^= x >> 31;
   return x;
}

static uint64_t
mixn(unsigned n, const uint64_t *v)
{
   uint64_t h = 0x1234567;
   for (unsigned i = 0; i < n; i++)
      h = mix(h, v[i]);
   return h;
}

struct lo_val {
   uint64_t v;
   unsigned width;
};

/* The value a read at position q of round round sees in loc. */
static struct lo_val
lo_read(const struct lo_eval *e, int loc, int64_t q, int round, bool all_in_round)
{
   const struct lo_write *best = NULL;
   int64_t best_key = 0;
   unsigned best_seq = 0;

   for (unsigned i = 0; i < e->num_writes; i++) {
      const struct lo_write *w = &e->writes[i];
      if (w->loc != loc)
         continue;
      bool earlier = w->round < round;
      bool visible = earlier || (all_in_round ? w->round == round
                                              : w->land <= q);
      if (!visible)
         continue;
      int64_t key = earlier ? w->issue - (1ll << 40) : w->land;
      if (!best || key > best_key || (key == best_key && w->seq > best_seq)) {
         best = w;
         best_key = key;
         best_seq = w->seq;
      }
   }
   if (!best)
      return (struct lo_val){mix(0xdead, loc), 1};
   return (struct lo_val){best->value, best->width};
}

static uint64_t
lo_read_half(const struct lo_eval *e, unsigned reg, unsigned half, int64_t q,
             int round, bool all)
{
   struct lo_val v = lo_read(e, reg * 2 + half, q, round, all);
   return v.width == 2 ? mix(mix(v.v, 0x4a1f), half) : v.v;
}

static uint64_t
lo_read_fp20(const struct lo_eval *e, unsigned reg, int64_t q, int round,
             bool all)
{
   struct lo_val l = lo_read(e, reg * 2, q, round, all);
   struct lo_val h = lo_read(e, reg * 2 + 1, q, round, all);
   if (l.v == h.v && l.width == 2 && h.width == 2)
      return l.v;
   uint64_t a[] = {l.v, h.v, l.width, h.width, 0x31d};
   return mixn(5, a);
}

static void
lo_write(struct lo_eval *e, int loc, uint64_t v, unsigned width,
         int64_t issue, int64_t land, int round)
{
   if (e->num_writes >= LO_MAX_WRITES) {
      e->why = "too many writes";
      return;
   }
   struct lo_write *w = &e->writes[e->num_writes];
   w->loc = loc;
   w->value = v;
   w->width = width;
   w->issue = issue;
   w->land = land;
   w->round = round;
   w->seq = e->num_writes++;
}

static uint64_t
lo_src(const struct lo_eval *e, const struct fp_alu_src_operand *s, int64_t q,
       int round)
{
   uint64_t leaf;
   unsigned reg = s->index;

   if (reg < 16) {
      leaf = s->datatype == FP_DATATYPE_FIXED10
                ? lo_read_half(e, reg, s->sub_reg_select_high, q, round, false)
                : lo_read_fp20(e, reg, q, round, false);
   } else if (reg >= 24 && reg <= 27) {
      leaf = 0; /* forwards: handled by the caller */
   } else {
      uint64_t a[] = {0x57a7, reg, s->datatype, s->sub_reg_select_high};
      leaf = mixn(4, a);
   }
   uint64_t a[] = {leaf, s->negate, s->absolute_value, s->scale_by_two,
                   s->minus_one, s->datatype, s->sub_reg_select_high};
   return mixn(7, a);
}

static bool
lo_supported_packet(struct lo_eval *e, const struct fp_alu_instr_packet *p)
{
   for (unsigned sl = 0; sl < 4; sl++) {
      const struct fp_alu_instr *a = &p->slots[sl];
      if (!a->dst.write_low_sub_reg && !a->dst.write_high_sub_reg)
         continue;
      if (a->dst.index >= 16 && a->dst.index != 31) {
         e->why = "ALU destination above R15";
         return false;
      }
      for (unsigned k = 0; k < 3; k++)
         if (a->src[k].index >= 24 && a->src[k].index <= 27) {
            e->why = "forward source";
            return false;
         }
   }
   return true;
}

bool
grate_fp_layout_signature(const struct grate_fp_shader *fp, bool sequential,
                          uint64_t *signature, const char **why)
{
   static struct lo_eval e_storage;
   struct lo_eval *e = &e_storage;
   e->sequential = sequential;
   e->rows = sequential ? 1 : MAX2(fp->info.num_rows, 1);
   e->num_writes = 0;
   e->why = NULL;
   *why = NULL;

   unsigned num_packets = list_length(&fp->alu_instructions);
   unsigned num_mfus = list_length(&fp->mfu_instructions);
   const struct fp_alu_instr_packet **pk =
      calloc(num_packets + 1, sizeof(*pk));
   const struct fp_mfu_instr **mf = calloc(num_mfus + 1, sizeof(*mf));
   if (!pk || !mf) {
      free(pk);
      free(mf);
      *why = "out of memory";
      return false;
   }
   unsigned i = 0;
   list_for_each_entry (struct fp_alu_instr_packet, p, &fp->alu_instructions,
                        link)
      pk[i++] = p;
   i = 0;
   list_for_each_entry (struct fp_mfu_instr, m, &fp->mfu_instructions, link)
      mf[i++] = m;

   uint64_t sig = 0x5167;
   int64_t g = 0;
   unsigned step = 0;
   unsigned seq_dw = 0;
   struct {
      const struct fp_instr *instr;
      unsigned base;
      int round;
      int64_t q;
      unsigned step;
   } dw[4];
   unsigned num_dw = 0, done_dw = 0;

   list_for_each_entry (struct fp_instr, instr, &fp->fp_instructions, link) {
      unsigned rows = e->rows;
      unsigned row = sequential ? 0 : step % rows;
      int round = sequential ? 0 : step / rows;
      unsigned base = row * 4;
      int64_t gfirst = g;

      if (instr->pseq || instr->spill_load || instr->dw.spill) {
         e->why = "spill or sequencer command";
         break;
      }

      if (instr->mfu_sched.num_instructions) {
         const struct fp_mfu_instr *m = mf[instr->mfu_sched.address];
         for (unsigned k = 0; k < 2; k++)
            if (m->mul[k].dst != 0 &&
                m->mul[k].dst != FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT) {
               e->why = "MFU multiplier writes a row register";
               break;
            }
         if (e->why)
            break;
         if (m->sfu.op != FP_SFU_OP_NOP && m->sfu.reg < 4) {
            uint64_t v = lo_read_fp20(e, base + m->sfu.reg, gfirst, round, false);
            uint64_t a[] = {0x5f0, m->sfu.op, v};
            uint64_t r = mixn(3, a);
            lo_write(e, (base + m->sfu.reg) * 2, r, 2, gfirst, gfirst, round);
            lo_write(e, (base + m->sfu.reg) * 2 + 1, r, 2, gfirst, gfirst,
                     round);
         } else if (m->sfu.op != FP_SFU_OP_NOP && m->sfu.reg > 4) {
            e->why = "SFU register above 4";
            break;
         }
         for (unsigned k = 0; k < 4; k++) {
            if (m->var[k].op == FP_VAR_OP_NOP)
               continue;
            if (m->var[k].op != FP_VAR_OP_FP20) {
               e->why = "FX10 interpolation";
               break;
            }
            uint64_t a[] = {0x1a7e, m->var[k].tram_row, k};
            uint64_t r = mixn(3, a);
            if (getenv("GRATE_LAYOUT_TRACE"))
               fprintf(stderr, "  %s step %u interp var%u row %u = %08x\n",
                       sequential ? "SEQ" : "HW ", step, k, row, (unsigned)r);
            lo_write(e, (base + k) * 2, r, 2, gfirst, gfirst, round);
            lo_write(e, (base + k) * 2 + 1, r, 2, gfirst, gfirst, round);
         }
         if (e->why)
            break;
      }

      if (instr->tex & (1 << 10)) {
         if (instr->tex & 0x10) {
            e->why = "fetch with the R2R3R0 coordinate source";
            break;
         }
         uint64_t x = lo_read_fp20(e, base + 0, gfirst, round, false);
         uint64_t y = lo_read_fp20(e, base + 1, gfirst, round, false);
         if (getenv("GRATE_LAYOUT_TRACE"))
            fprintf(stderr, "  %s step %u TEX row %u coords %08x %08x\n",
                    sequential ? "SEQ" : "HW ", step, row, (unsigned)x, (unsigned)y);
         for (unsigned c = 0; c < 4; c++) {
            uint64_t a[] = {0x7e5, instr->tex & 0x3ff, x, y, c};
            lo_write(e, (base + 2 + c / 2) * 2 + c % 2, mixn(5, a), 1, gfirst,
                     gfirst, round);
         }
      }

      for (unsigned p = 0; p < instr->alu_sched.num_instructions; p++) {
         unsigned address = instr->alu_sched.address + p;
         if (address >= num_packets) {
            e->why = "packet address out of range";
            break;
         }
         const struct fp_alu_instr_packet *packet = pk[address];
         if (!lo_supported_packet(e, packet))
            break;
         struct {
            int loc[2];
            unsigned n;
            uint64_t v;
            unsigned width;
         } res[4];
         unsigned nres = 0;
         for (unsigned sl = 0; sl < 4; sl++) {
            const struct fp_alu_instr *a = &packet->slots[sl];
            if (!a->dst.write_low_sub_reg && !a->dst.write_high_sub_reg)
               continue;
            uint64_t h[12];
            unsigned n = 0;
            h[n++] = a->op;
            h[n++] = a->scale;
            h[n++] = a->dst.saturate;
            h[n++] = a->condition;
            h[n++] = a->dst.write_low_sub_reg | (a->dst.write_high_sub_reg << 1);
            for (unsigned k = 0; k < 3; k++)
               h[n++] = lo_src(e, &a->src[k], g, round);
            if (a->condition != FP_CONDITION_ALWAYS && a->dst.index < 16) {
               if (a->dst.write_low_sub_reg)
                  h[n++] = lo_read_half(e, a->dst.index, 0, g, round, false);
               if (a->dst.write_high_sub_reg)
                  h[n++] = lo_read_half(e, a->dst.index, 1, g, round, false);
            }
            uint64_t v = mixn(n, h);
            if (getenv("GRATE_LAYOUT_TRACE"))
               fprintf(stderr, "  %s step %u g %lld slot %u -> r%u%s%s = %08x  (A %08x B %08x C %08x)\n",
                       sequential ? "SEQ" : "HW ", step, (long long)g, sl,
                       a->dst.index, a->dst.write_low_sub_reg ? "l" : "",
                       a->dst.write_high_sub_reg ? "h" : "", (unsigned)v,
                       (unsigned)h[5], (unsigned)h[6], (unsigned)h[7]);
            if (a->dst.index == 31)
               continue;
            res[nres].n = 0;
            if (a->dst.write_low_sub_reg)
               res[nres].loc[res[nres].n++] = a->dst.index * 2;
            if (a->dst.write_high_sub_reg)
               res[nres].loc[res[nres].n++] = a->dst.index * 2 + 1;
            res[nres].v = v;
            res[nres].width = res[nres].n;
            nres++;
         }
         for (unsigned r = 0; r < nres; r++)
            for (unsigned k = 0; k < res[r].n; k++)
               lo_write(e, res[r].loc[k], res[r].v, res[r].width, g,
                        sequential ? g + 1 : g + 2, round);
         g++;
      }
      if (e->why)
         break;

      if (instr->dw.enable) {
         if (num_dw >= 4) {
            e->why = "too many output writes";
            break;
         }
         dw[num_dw].instr = instr;
         dw[num_dw].base = base;
         dw[num_dw].round = round;
         dw[num_dw].q = sequential ? g : gfirst;
         dw[num_dw].step = step;
         num_dw++;
      }
      /* The output write fires at the end of its round (at once when the
       * steps are evaluated one after the other). */
      bool round_ends = sequential || (step + 1) % e->rows == 0 ||
                        instr->link.next == &fp->fp_instructions;
      while (round_ends && done_dw < num_dw) {
         const struct fp_instr *d = dw[done_dw].instr;
         unsigned regs = d->dw.src_regs == FP_DW_REGS_R2_R3 ? 2 : 0;
         uint64_t h[8];
         unsigned n = 0;
         h[n++] = 0xd7;
         h[n++] = d->dw.index;
         h[n++] = d->dw.stencil_write;
         h[n++] = regs;
         for (unsigned r = 0; r < 2; r++)
            for (unsigned half = 0; half < 2; half++)
               h[n++] = lo_read_half(e, dw[done_dw].base + regs + r, half,
                                     dw[done_dw].q, dw[done_dw].round, true);
         if (getenv("GRATE_LAYOUT_TRACE"))
            fprintf(stderr, "  %s step %u DW inputs %08x %08x %08x %08x\n",
                    sequential ? "SEQ" : "HW ", dw[done_dw].step,
                    (unsigned)h[4], (unsigned)h[5], (unsigned)h[6],
                    (unsigned)h[7]);
         sig = mix(sig, mixn(n, h));
         seq_dw++;
         done_dw++;
      }
      step++;
   }

   free(pk);
   free(mf);
   if (e->why) {
      *why = e->why;
      return false;
   }
   if (!seq_dw) {
      *why = "no output write";
      return false;
   }
   *signature = sig;
   return true;
}
