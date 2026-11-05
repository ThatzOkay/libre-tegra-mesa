#ifndef GRATE_FP_BLEND_H
#define GRATE_FP_BLEND_H

#include <stdint.h>

#include "pipe/p_state.h"

#include "grate_compiler.h"

/* Blend epilogue of the fragment program, see blend.c. */

#define GRATE_FP_BLEND_MAX_PACKETS 3

/**
 * Packs the blend state of render target 0 into a 32-bit key that names one
 * epilogue. Returns 0 when nothing needs doing (blending off and every
 * channel written), or when the state uses an equation or factor the
 * epilogue cannot express yet.
 */
uint32_t grate_fp_blend_key(const struct pipe_rt_blend_state *rt);

/**
 * Builds the ALU packets of the epilogue for a key from
 * grate_fp_blend_key(). Writes 8 words per packet to words and returns the
 * number of packets (at most GRATE_FP_BLEND_MAX_PACKETS), or 0 for a key
 * that is not valid.
 */
unsigned grate_fp_blend_alu(uint32_t key,
                            uint32_t words[GRATE_FP_BLEND_MAX_PACKETS * 8]);

/**
 * Dual-source blending (GL_EXT_blend_func_extended), where factors come from
 * the fragment program's second output, src1:
 *
 *    out = src0 * Fs + dst * Fd         with Fs, Fd made of src1
 *
 * The epilogue cannot hold src0, src1 and the destination at once (a
 * per-pixel value in a global register does not survive the destination
 * read, and R0-R3 hold eight FX10 values), so the draw is split in
 * passes that each blend one colour, which is exact as long as the
 * fragments of one draw do not overlap:
 *
 *    pass dst:  colour = src1,           blend ZERO, Fd'      -> dst * Fd
 *    pass src:  colour = src0 * Fs(src1), blend ONE, ONE      -> + src0 * Fs
 *
 * where Fd' is Fd written with the colour (SRC1_COLOR -> SRC_COLOR).
 */
#define GRATE_FP_DUAL_MAX_PASSES 2

struct grate_fp_dual_pass {
   struct grate_fp_output output; /* what the fragment program stores */
   struct pipe_rt_blend_state rt; /* how that colour is blended */
};

struct grate_fp_dual_plan {
   unsigned num_passes;
   struct grate_fp_dual_pass pass[GRATE_FP_DUAL_MAX_PASSES];
};

/** True if a factor of the state comes from src1. */
bool grate_fp_blend_is_dual(const struct pipe_rt_blend_state *rt);

/**
 * Splits a dual-source state of render target 0 in passes. Returns false
 * for a combination that has no such split (the caller logs the state).
 */
bool grate_fp_blend_dual_plan(const struct pipe_rt_blend_state *rt,
                              struct grate_fp_dual_plan *plan);

#endif
