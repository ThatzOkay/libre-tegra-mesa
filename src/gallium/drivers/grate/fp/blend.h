#ifndef GRATE_FP_BLEND_H
#define GRATE_FP_BLEND_H

#include <stdint.h>

#include "pipe/p_state.h"

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

#endif
