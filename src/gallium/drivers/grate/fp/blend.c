#include <string.h>

#include "util/macros.h"

#include "blend.h"
#include "fpir.h"

/*
 * The 3D unit has no blender. A blended fragment program is the application's
 * program plus one more sequencer instruction that reads the destination
 * pixel into R0/R1, and a few ALU packets that combine it with the colour the
 * application left in R2/R3. Both hold FX10 values, two per register.
 *
 * The packets work on storage positions, not channels:
 *
 *    position  0    1    2    3
 *    channel   B    G    R    A
 *    register  R2   R2   R3   R3     (low, high, low, high half)
 *
 * and the same position in R0/R1 is the destination value. One ALU slot
 * computes A * B + C, and optionally + C * D, but D has to be a half of the
 * same register as C. That gives two shapes per position:
 *
 *    plain (source factor ONE or ZERO):
 *       out = dst * Fd + src                       one packet
 *    general:
 *       G = src * Fs  (into global register 16 + position)
 *       out = dst * Fd + G                         packet, nop, packet
 *
 * The nop is the latency gap the compiler's own output stage leaves between
 * a computation and the instruction that stores it.
 *
 * To extend this:
 *
 *  - A factor built from values that can already be read (for example
 *    INV_DST_ALPHA) is one row in blend_factors[].
 *  - A new value (the blend constant from a uniform, the second fragment
 *    output of dual-source blending) is a member of enum blend_value and a
 *    case in blend_value_operand(); the epilogue also has to make that value
 *    readable, which grate_program.c does by placing it in a register that
 *    survives the destination read.
 *  - A new equation is a row in blend_funcs[] if it is a signed sum of the
 *    two products. MIN and MAX are different ALU ops and need their own
 *    branch in grate_fp_blend_alu().
 *
 * Whatever is not in a table makes grate_fp_blend_key() return 0, and the
 * caller logs the complete state, so nothing is drawn wrongly without a
 * trace.
 */

enum blend_value {
   BLEND_VALUE_ZERO,
   BLEND_VALUE_ONE,
   BLEND_VALUE_SRC_COLOR,
   BLEND_VALUE_SRC_ALPHA,
   BLEND_VALUE_DST_COLOR,
   BLEND_VALUE_DST_ALPHA,
};

static const struct {
   enum pipe_blendfactor factor;
   enum blend_value value;
   bool invert; /* the factor is 1 - value */
} blend_factors[] = {
   {PIPE_BLENDFACTOR_ZERO, BLEND_VALUE_ZERO, false},
   {PIPE_BLENDFACTOR_ONE, BLEND_VALUE_ONE, false},
   {PIPE_BLENDFACTOR_SRC_COLOR, BLEND_VALUE_SRC_COLOR, false},
   {PIPE_BLENDFACTOR_INV_SRC_COLOR, BLEND_VALUE_SRC_COLOR, true},
   {PIPE_BLENDFACTOR_SRC_ALPHA, BLEND_VALUE_SRC_ALPHA, false},
   {PIPE_BLENDFACTOR_INV_SRC_ALPHA, BLEND_VALUE_SRC_ALPHA, true},
   {PIPE_BLENDFACTOR_DST_COLOR, BLEND_VALUE_DST_COLOR, false},
   {PIPE_BLENDFACTOR_INV_DST_COLOR, BLEND_VALUE_DST_COLOR, true},
   {PIPE_BLENDFACTOR_DST_ALPHA, BLEND_VALUE_DST_ALPHA, false},
   {PIPE_BLENDFACTOR_INV_DST_ALPHA, BLEND_VALUE_DST_ALPHA, true},
};

static const struct {
   enum pipe_blend_func func;
   bool negate_dst_term; /* result = src * Fs - dst * Fd */
   bool negate_src_term; /* result = dst * Fd - src * Fs */
} blend_funcs[] = {
   {PIPE_BLEND_ADD, false, false},
   {PIPE_BLEND_SUBTRACT, true, false},
   {PIPE_BLEND_REVERSE_SUBTRACT, false, true},
};

/* Bit layout of the key. */
#define KEY_VALID            (1u << 31)
#define KEY_RGB_FUNC_SHIFT   0
#define KEY_ALPHA_FUNC_SHIFT 3
#define KEY_RGB_SRC_SHIFT    6
#define KEY_RGB_DST_SHIFT    11
#define KEY_ALPHA_SRC_SHIFT  16
#define KEY_ALPHA_DST_SHIFT  21
#define KEY_COLORMASK_SHIFT  26
#define KEY_FUNC_MASK        0x7
#define KEY_FACTOR_MASK      0x1f
#define KEY_COLORMASK_MASK   0xf

/* The colormask bit of each storage position: B, G, R, A. */
static const unsigned position_mask_bit[4] = {0x4, 0x2, 0x1, 0x8};

static struct fp_alu_src_operand
src_zero(void)
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
   };

   return src;
}

static struct fp_alu_src_operand
src_one(void)
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = true,
   };

   return src;
}

/* Where a blend value of storage position p is read from. */
static struct fp_alu_src_operand
blend_value_operand(enum blend_value value, unsigned p)
{
   struct fp_alu_src_operand src = {.datatype = FP_DATATYPE_FIXED10};

   switch (value) {
   case BLEND_VALUE_ZERO:
      return src_zero();
   case BLEND_VALUE_ONE:
      return src_one();
   case BLEND_VALUE_SRC_COLOR:
      src.index = 2 + p / 2;
      src.sub_reg_select_high = p & 1;
      return src;
   case BLEND_VALUE_SRC_ALPHA:
      src.index = 3;
      src.sub_reg_select_high = true;
      return src;
   case BLEND_VALUE_DST_COLOR:
      src.index = p / 2;
      src.sub_reg_select_high = p & 1;
      return src;
   case BLEND_VALUE_DST_ALPHA:
      src.index = 1;
      src.sub_reg_select_high = true;
      return src;
   }
   UNREACHABLE("invalid blend value");
}

/* Operand for a blend factor; false if the factor is not in the table. */
static bool
blend_factor_operand(enum pipe_blendfactor factor, unsigned p,
                     struct fp_alu_src_operand *src)
{
   for (unsigned i = 0; i < ARRAY_SIZE(blend_factors); i++) {
      if (blend_factors[i].factor != factor)
         continue;

      *src = blend_value_operand(blend_factors[i].value, p);
      if (blend_factors[i].invert) {
         /* 1 - x, as the premultiplied epilogue always did it */
         src->minus_one = true;
         src->negate = true;
      }
      return true;
   }
   return false;
}

static bool
blend_func_supported(enum pipe_blend_func func)
{
   for (unsigned i = 0; i < ARRAY_SIZE(blend_funcs); i++) {
      if (blend_funcs[i].func == func)
         return true;
   }
   return false;
}

uint32_t
grate_fp_blend_key(const struct pipe_rt_blend_state *rt)
{
   struct fp_alu_src_operand probe;

   /* Without blending only the colormask matters, and only when it
    * protects a channel: that is "src * 1 + dst * 0" with the protected
    * channels kept.
    */
   if (!rt->blend_enable) {
      if ((rt->colormask & KEY_COLORMASK_MASK) == KEY_COLORMASK_MASK)
         return 0;

      return KEY_VALID | PIPE_BLEND_ADD << KEY_RGB_FUNC_SHIFT |
             PIPE_BLEND_ADD << KEY_ALPHA_FUNC_SHIFT |
             PIPE_BLENDFACTOR_ONE << KEY_RGB_SRC_SHIFT |
             PIPE_BLENDFACTOR_ZERO << KEY_RGB_DST_SHIFT |
             PIPE_BLENDFACTOR_ONE << KEY_ALPHA_SRC_SHIFT |
             PIPE_BLENDFACTOR_ZERO << KEY_ALPHA_DST_SHIFT |
             (rt->colormask & KEY_COLORMASK_MASK) << KEY_COLORMASK_SHIFT;
   }

   if (!blend_func_supported(rt->rgb_func) ||
       !blend_func_supported(rt->alpha_func))
      return 0;

   for (unsigned p = 0; p < 4; p++) {
      bool alpha = p == 3;
      enum pipe_blendfactor src =
         alpha ? rt->alpha_src_factor : rt->rgb_src_factor;
      enum pipe_blendfactor dst =
         alpha ? rt->alpha_dst_factor : rt->rgb_dst_factor;

      if (!blend_factor_operand(src, p, &probe) ||
          !blend_factor_operand(dst, p, &probe))
         return 0;
   }

   return KEY_VALID | rt->rgb_func << KEY_RGB_FUNC_SHIFT |
          rt->alpha_func << KEY_ALPHA_FUNC_SHIFT |
          rt->rgb_src_factor << KEY_RGB_SRC_SHIFT |
          rt->rgb_dst_factor << KEY_RGB_DST_SHIFT |
          rt->alpha_src_factor << KEY_ALPHA_SRC_SHIFT |
          rt->alpha_dst_factor << KEY_ALPHA_DST_SHIFT |
          (rt->colormask & KEY_COLORMASK_MASK) << KEY_COLORMASK_SHIFT;
}

unsigned
grate_fp_blend_alu(uint32_t key,
                   uint32_t words[GRATE_FP_BLEND_MAX_PACKETS * 8])
{
   struct fp_alu_instr packets[GRATE_FP_BLEND_MAX_PACKETS][4] = {{{0}}};
   struct fp_alu_src_operand fs[4], fd[4];
   enum pipe_blend_func func[4];
   bool general = false;
   unsigned num_packets;
   unsigned colormask = (key >> KEY_COLORMASK_SHIFT) & KEY_COLORMASK_MASK;

   if (!(key & KEY_VALID))
      return 0;

   for (unsigned p = 0; p < 4; p++) {
      bool alpha = p == 3;
      enum pipe_blendfactor src_factor =
         (key >> (alpha ? KEY_ALPHA_SRC_SHIFT : KEY_RGB_SRC_SHIFT)) &
         KEY_FACTOR_MASK;
      enum pipe_blendfactor dst_factor =
         (key >> (alpha ? KEY_ALPHA_DST_SHIFT : KEY_RGB_DST_SHIFT)) &
         KEY_FACTOR_MASK;

      func[p] = (key >> (alpha ? KEY_ALPHA_FUNC_SHIFT : KEY_RGB_FUNC_SHIFT)) &
                KEY_FUNC_MASK;

      /* A channel the colormask protects keeps the destination. */
      if (!(colormask & position_mask_bit[p])) {
         src_factor = PIPE_BLENDFACTOR_ZERO;
         dst_factor = PIPE_BLENDFACTOR_ONE;
         func[p] = PIPE_BLEND_ADD;
      }

      if (!blend_factor_operand(src_factor, p, &fs[p]) ||
          !blend_factor_operand(dst_factor, p, &fd[p]))
         return 0;

      general |= src_factor != PIPE_BLENDFACTOR_ONE &&
                 src_factor != PIPE_BLENDFACTOR_ZERO;
   }

   if (general) {
      for (unsigned p = 0; p < 4; p++) {
         /* G = src * Fs */
         packets[0][p] = (struct fp_alu_instr){
            .op = FP_ALU_OP_MAD,
            .dst =
               {
                  .index = 16 + p,
                  .write_low_sub_reg = true,
                  .write_high_sub_reg = true,
               },
            .src =
               {
                  blend_value_operand(BLEND_VALUE_SRC_COLOR, p),
                  fs[p],
                  src_zero(),
                  {.index = 0, .datatype = FP_DATATYPE_FIXED10},
               },
         };

         /* nop */
         packets[1][p] = (struct fp_alu_instr){
            .op = FP_ALU_OP_MAD,
            .dst = {.index = 31},
            .src = {src_zero(), src_one(), src_zero(), src_one()},
         };
      }
      num_packets = 3;
   } else {
      num_packets = 1;
   }

   for (unsigned p = 0; p < 4; p++) {
      struct fp_alu_src_operand dst_term =
         blend_value_operand(BLEND_VALUE_DST_COLOR, p);
      struct fp_alu_src_operand src_term;
      struct fp_alu_instr *out = &packets[num_packets - 1][p];

      if (general) {
         src_term = (struct fp_alu_src_operand){
            .index = 16 + p,
            .datatype = FP_DATATYPE_FP20,
         };
      } else if (fs[p].index == 31 && !fs[p].sub_reg_select_high) {
         src_term = src_zero();
      } else {
         src_term = blend_value_operand(BLEND_VALUE_SRC_COLOR, p);
      }

      for (unsigned i = 0; i < ARRAY_SIZE(blend_funcs); i++) {
         if (blend_funcs[i].func != func[p])
            continue;
         dst_term.negate = blend_funcs[i].negate_dst_term;
         src_term.negate = blend_funcs[i].negate_src_term;
      }

      *out = (struct fp_alu_instr){
         .op = FP_ALU_OP_MAD,
         .dst =
            {
               .index = 2 + p / 2,
               .write_low_sub_reg = !(p & 1),
               .write_high_sub_reg = p & 1,
               .saturate = true,
            },
         .src = {dst_term, fd[p], src_term, src_one()},
      };

      /* An rD on the same register as rC switches on the second product
       * (see grate_fp_pack_alu()).
       */
      if (out->src[2].index == out->src[3].index)
         out->src[3].index = 0;
   }

   for (unsigned i = 0; i < num_packets; i++) {
      for (unsigned p = 0; p < 4; p++)
         grate_fp_pack_alu(words + i * 8 + p * 2, &packets[i][p]);
   }

   return num_packets;
}

static bool
factor_is_dual(enum pipe_blendfactor factor)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_SRC1_COLOR:
   case PIPE_BLENDFACTOR_SRC1_ALPHA:
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      return true;
   default:
      return false;
   }
}

bool
grate_fp_blend_is_dual(const struct pipe_rt_blend_state *rt)
{
   return rt->blend_enable && (factor_is_dual(rt->rgb_src_factor) ||
                               factor_is_dual(rt->alpha_src_factor) ||
                               factor_is_dual(rt->rgb_dst_factor) ||
                               factor_is_dual(rt->alpha_dst_factor));
}

/* The factors a pass may use, one row each. `colour` and `alpha` are how
 * the factor reads when the colour channels / the alpha channel are blended
 * against the stored colour (src1 is stored as the colour in the dst pass).
 */
static const struct {
   enum pipe_blendfactor factor;
   enum grate_fp_dual_factor dual;
   enum pipe_blendfactor colour;
   enum pipe_blendfactor alpha;
} dual_factors[] = {
   {PIPE_BLENDFACTOR_ZERO, GRATE_FP_DUAL_ZERO, PIPE_BLENDFACTOR_ZERO,
    PIPE_BLENDFACTOR_ZERO},
   {PIPE_BLENDFACTOR_ONE, GRATE_FP_DUAL_ONE, PIPE_BLENDFACTOR_ONE,
    PIPE_BLENDFACTOR_ONE},
   {PIPE_BLENDFACTOR_SRC1_COLOR, GRATE_FP_DUAL_SRC1_COLOR,
    PIPE_BLENDFACTOR_SRC_COLOR, PIPE_BLENDFACTOR_SRC_ALPHA},
   {PIPE_BLENDFACTOR_INV_SRC1_COLOR, GRATE_FP_DUAL_INV_SRC1_COLOR,
    PIPE_BLENDFACTOR_INV_SRC_COLOR, PIPE_BLENDFACTOR_INV_SRC_ALPHA},
   {PIPE_BLENDFACTOR_SRC1_ALPHA, GRATE_FP_DUAL_SRC1_ALPHA,
    PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_SRC_ALPHA},
   {PIPE_BLENDFACTOR_INV_SRC1_ALPHA, GRATE_FP_DUAL_INV_SRC1_ALPHA,
    PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA},
};

static bool
dual_factor_lookup(enum pipe_blendfactor factor, unsigned *index)
{
   for (unsigned i = 0; i < ARRAY_SIZE(dual_factors); i++) {
      if (dual_factors[i].factor == factor) {
         *index = i;
         return true;
      }
   }
   return false;
}

bool
grate_fp_blend_dual_plan(const struct pipe_rt_blend_state *rt,
                         struct grate_fp_dual_plan *plan)
{
   unsigned src_rgb, src_alpha, dst_rgb, dst_alpha;

   memset(plan, 0, sizeof(*plan));

   /* Only sums: both passes add their term. */
   if (rt->rgb_func != PIPE_BLEND_ADD || rt->alpha_func != PIPE_BLEND_ADD)
      return false;

   if (!dual_factor_lookup(rt->rgb_src_factor, &src_rgb) ||
       !dual_factor_lookup(rt->alpha_src_factor, &src_alpha) ||
       !dual_factor_lookup(rt->rgb_dst_factor, &dst_rgb) ||
       !dual_factor_lookup(rt->alpha_dst_factor, &dst_alpha))
      return false;

   /* dst * Fd, unless Fd is one */
   if (dual_factors[dst_rgb].factor != PIPE_BLENDFACTOR_ONE ||
       dual_factors[dst_alpha].factor != PIPE_BLENDFACTOR_ONE) {
      struct grate_fp_dual_pass *pass = &plan->pass[plan->num_passes++];

      pass->output.mode = GRATE_FP_OUTPUT_SRC1;
      pass->rt = *rt;
      pass->rt.rgb_func = pass->rt.alpha_func = PIPE_BLEND_ADD;
      pass->rt.rgb_src_factor = pass->rt.alpha_src_factor =
         PIPE_BLENDFACTOR_ZERO;
      pass->rt.rgb_dst_factor = dual_factors[dst_rgb].colour;
      pass->rt.alpha_dst_factor = dual_factors[dst_alpha].alpha;
   }

   /* + src0 * Fs, unless Fs is zero */
   if (dual_factors[src_rgb].factor != PIPE_BLENDFACTOR_ZERO ||
       dual_factors[src_alpha].factor != PIPE_BLENDFACTOR_ZERO) {
      struct grate_fp_dual_pass *pass = &plan->pass[plan->num_passes++];

      pass->output.mode = GRATE_FP_OUTPUT_SRC0_TIMES_FACTOR;
      pass->output.factor[0] = dual_factors[src_rgb].dual;
      pass->output.factor[1] = dual_factors[src_alpha].dual;
      pass->rt = *rt;
      pass->rt.rgb_func = pass->rt.alpha_func = PIPE_BLEND_ADD;
      pass->rt.rgb_src_factor = pass->rt.alpha_src_factor =
         PIPE_BLENDFACTOR_ONE;
      pass->rt.rgb_dst_factor = pass->rt.alpha_dst_factor =
         PIPE_BLENDFACTOR_ONE;
   }

   return true;
}
