#ifndef GRATE_COMPILER_H
#define GRATE_COMPILER_H

#include "util/list.h"

#include <stdint.h>
#include <stdbool.h>

struct tgsi_parse_context;

/*
 * Constant RAM layout of a vertex program. Immediates (TGSI_FILE_IMMEDIATE) come first: immediate i is vec4 slot i. The
 * application's uniforms (TGSI_FILE_CONSTANT) follow: uniform k is vec4 slot num_immediates + k. Both are uploaded as one
 * block starting at slot 0 (see emit_vs_uniforms()). A write to REG_TGR3D_VPE_CONST_OFFSET with a non-zero index was
 * found not to move the upload position (2026-10-31, on hardware: data written after it still landed in slot 0), so
 * everything must start at offset 0, and the layout has to be known when the program is compiled; this is why the
 * immediates are put first and the uniforms moved up, and not the other way round.
 */
#define GRATE_VP_MAX_IMMEDIATES 32

struct grate_vp_shader {
   struct list_head instructions;
   uint16_t output_mask;
   uint32_t immediates[GRATE_VP_MAX_IMMEDIATES][4]; /* raw 32-bit words, slot i of the immediates */
   unsigned num_immediates;
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
   } constants[32];
   unsigned num_constants;
};

#define GRATE_FP_MAX_IMMEDIATES 32
#define GRATE_FP_MAX_TEMPS 64

struct grate_fp_shader {
   unsigned num_temporaries;
   /* Encoded scalar locations: register[6:0], FX10 bit 8, high half bit 9.
    * The extra virtual temporary is reserved for interpolation staging. */
   int temp_src[GRATE_FP_MAX_TEMPS + 1][4];
   int temp_dst[GRATE_FP_MAX_TEMPS + 1][4];
   uint8_t occupied[72];
   uint8_t claimed[72];
   uint8_t reusable[72];
   bool late_rows;
   bool texture_rows;
   bool allocation_failed;
   float immediates[GRATE_FP_MAX_IMMEDIATES][4];
   unsigned num_immediates;
   struct list_head fp_instructions;
   struct list_head alu_instructions;
   struct list_head mfu_instructions;
   struct grate_fp_info info;
};

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi);

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi);

#endif
