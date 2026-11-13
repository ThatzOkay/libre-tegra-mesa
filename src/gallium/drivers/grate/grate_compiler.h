#ifndef GRATE_COMPILER_H
#define GRATE_COMPILER_H

#include "util/list.h"

#include <stdint.h>
#include <stdbool.h>

struct tgsi_parse_context;
struct tgsi_full_instruction;

/*
 * Constant RAM layout of a vertex program. Immediates (TGSI_FILE_IMMEDIATE)
 * come first: immediate i is vec4 slot i. The application's uniforms
 * (TGSI_FILE_CONSTANT) follow: uniform k is vec4 slot num_immediates + k.
 * Both are uploaded as one block starting at slot 0 (see emit_vs_uniforms()).
 * A write to REG_TGR3D_VPE_CONST_OFFSET with a non-zero index was found not
 * to move the upload position (2026-10-31, on hardware: data written after it
 * still landed in slot 0), so everything must start at offset 0, and the
 * layout has to be known when the program is compiled; this is why the
 * immediates are put first and the uniforms moved up, and not the other way
 * round.
 */
#define GRATE_VP_MAX_IMMEDIATES 32

/* Logical uniforms are compacted into the scalar FP20 constant bank. */
#define GRATE_FP_MAX_CONSTANTS 32
/* Constants a program may use before it is split into epochs of at most
 * GRATE_FP_MAX_CONSTANTS each. */
#define GRATE_FP_MAX_LOGICAL_CONSTANTS 192
#define GRATE_FP_UNIFORM_COMPONENTS 256

struct grate_vp_shader {
   struct list_head instructions;
   uint16_t output_mask;
   uint32_t immediates[GRATE_VP_MAX_IMMEDIATES]
                      [4]; /* raw 32-bit words, slot i of the immediates */
   unsigned num_immediates;
   unsigned num_temps; /* TGSI temporaries declared; scratch temporaries start
                          here */
   bool failed; /* the program cannot be compiled: refuse the shader */
};

struct grate_bo;

/* A fragment program of more than one epoch, in NVIDIA's scheme: every
 * epoch is a block of register writes (the program of up to 64 steps, the
 * resets of the instruction offsets and the like) that the sequencer loads
 * by DMA from a program buffer, plus a packet of fragment constants that it
 * loads by DMA from a constant buffer, and the epochs run one after the
 * other with their accumulators kept between them. The program buffer holds
 * the blocks; the constant buffer is rewritten for every draw from the
 * uniforms. */
#define GRATE_FP_MAX_EPOCHS 5

struct grate_fp_epoch {
   unsigned prog_offset; /* byte offset of the block in the program buffer */
   unsigned prog_words;  /* length of the block in words */
   unsigned steps;       /* instructions of this epoch */
   unsigned packets;     /* ALU packets of this epoch */
   unsigned const_bank;  /* first bank slot of the constant packet */
   unsigned num_constants;
   struct {
      int word; /* -1 for an immediate */
      float immediate;
   } constants[GRATE_FP_MAX_CONSTANTS];
};

struct grate_fp_epochs {
   unsigned count;
   struct grate_fp_epoch epoch[GRATE_FP_MAX_EPOCHS];
   uint32_t *program; /* the blocks, uploaded on first use */
   unsigned program_bytes;
   struct grate_bo *program_bo;
   struct grate_bo *constant_bo;
   unsigned constant_bytes;
};

struct grate_fp_info {
   struct {
      uint32_t src;
      uint32_t dst;
   } inputs[16];
   int num_inputs;
   int color_input;
   int max_tram_row;
   /* Scalar hardware constants: uniforms point into TGSI constant buffer 0;
    * immediates share the same 32-entry FP20 bank. */
   struct {
      int word; /* -1 for an immediate */
      float immediate;
   } constants[GRATE_FP_MAX_LOGICAL_CONSTANTS];
   unsigned num_constants;
   /* Register rows (groups of four registers R4k..R4k+3) the program needs,
    * 1 to 4: GLOBAL_PIX_ATTR.NUM_ROWS + 1. Set by grate_tgsi_to_fp(). */
   unsigned num_rows;
   unsigned num_spill_slots;
   /* Not NULL for a program of more than one epoch, which owns it. */
   struct grate_fp_epochs *epochs;
};

/* NUM_QID - 1 of the register file for a program of 1 to 4 register rows,
 * the numbers of NVIDIA's driver for a capacity of 0x12c. */
static inline unsigned
grate_fp_max_qid(unsigned rows)
{
   static const unsigned max_qid_for_rows[] = {0, (0x12c - 1) >> 2,
                                               (0x12c - 1) >> 3,
                                               (0x12c - 1) / 12,
                                               (0x12c - 1) >> 4};
   return max_qid_for_rows[rows];
}

#define GRATE_FP_MAX_IMMEDIATES 32
#define GRATE_FP_MAX_TEMPS      64
/* Temporaries that live within one TGSI instruction (staging of varyings and
 * scratch sums) come after the declared ones. */
#define GRATE_FP_STAGE_TEMPS    3
#define GRATE_FP_MAX_SPILLS     256
/* P128 elements, four pixels per quad, and 19 quad IDs at four rows. */
#define GRATE_FP_SPILL_SURFACE 4
#define GRATE_FP_SPILL_STRIDE  (16 * 4 * 19)

struct grate_fp_shader {
   unsigned num_temporaries;
   /* Encoded scalar locations: register[6:0], FX10 bit 8, high half bit 9;
    * -1 is absent and -2-slot names an FP20 scalar in the spill surface.
    * The extra virtual temporary is reserved for interpolation staging. */
   int temp_src[GRATE_FP_MAX_TEMPS + GRATE_FP_STAGE_TEMPS][4];
   int temp_dst[GRATE_FP_MAX_TEMPS + GRATE_FP_STAGE_TEMPS][4];
   /* A clean backing slot remains valid after a reload until this
    * virtual component is overwritten or dies. */
   int spill_slot[GRATE_FP_MAX_TEMPS + GRATE_FP_STAGE_TEMPS][4];
   uint8_t occupied[72];
   uint8_t claimed[72];
   uint8_t reusable[72];
   bool spilling;
   /* DDX/DDY use the quad exchange registers r32, r40, r48 and r56, which
    * the constants must not occupy. */
   bool uses_derivatives;
   bool register_pressure;
   const struct tgsi_full_instruction *current_instruction;
   const struct tgsi_full_instruction *code_end;
   uint8_t pinned[72];
   bool spill_used[GRATE_FP_MAX_SPILLS];
   bool spill_fx10[GRATE_FP_MAX_SPILLS];
   bool late_rows;
   bool texture_rows;
   /* the value being allocated lives across an SFU op, which needs rows R0-R3
    * for its operand and result: keep it out of them */
   bool avoid_rows;
   bool allocation_failed;
   float immediates[GRATE_FP_MAX_IMMEDIATES][4];
   unsigned num_immediates;
   struct list_head fp_instructions;
   struct list_head alu_instructions;
   struct list_head mfu_instructions;
   struct grate_fp_info info;
};

/**
 * Which fragment output a fragment program stores as its colour. Dual-source
 * blending (fp/blend.c) splits a draw into passes that each store one value
 * computed from the shader's two outputs, src0 (output 0) and src1
 * (output 1).
 */
enum grate_fp_output_mode {
   /* output 0, as the application wrote it */
   GRATE_FP_OUTPUT_SRC0,
   /* output 1 */
   GRATE_FP_OUTPUT_SRC1,
   /* output 0 * factor(output 1), see grate_fp_dual_factor */
   GRATE_FP_OUTPUT_SRC0_TIMES_FACTOR,
};

/* A blend factor made of src1 (the second fragment output). */
enum grate_fp_dual_factor {
   GRATE_FP_DUAL_ZERO,
   GRATE_FP_DUAL_ONE,
   GRATE_FP_DUAL_SRC1_COLOR,
   GRATE_FP_DUAL_INV_SRC1_COLOR,
   GRATE_FP_DUAL_SRC1_ALPHA,
   GRATE_FP_DUAL_INV_SRC1_ALPHA,
};

struct grate_fp_output {
   enum grate_fp_output_mode mode;
   /* GRATE_FP_OUTPUT_SRC0_TIMES_FACTOR: the factor of the colour channels
    * and of the alpha channel */
   enum grate_fp_dual_factor factor[2];
};

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi);

void grate_tgsi_to_fp(struct grate_fp_shader *fp,
                      struct tgsi_parse_context *tgsi,
                      const struct grate_fp_output *output, bool spilling);

#endif
