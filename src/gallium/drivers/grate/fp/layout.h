#ifndef GRATE_FP_LAYOUT_H
#define GRATE_FP_LAYOUT_H

#include <stdbool.h>
#include <stdint.h>

#include "util/list.h"

#include "grate_compiler.h"

/**
 * Symbolic evaluation of a fragment program's steps (see layout.c).
 *
 * Every ALU slot and every unit operation (interpolation, SFU, fetch) is an
 * uninterpreted function of the values it reads; the result is a hash of what
 * the output write sees. With sequential = true the steps are evaluated the
 * way the emitter wrote them (one step after the other, a result visible to
 * the next packet, all units on row 0). With sequential = false they are
 * evaluated with the hardware rules measured in docs/grate-alu-latency.md
 * (write latency of two packets inside a round, unit operations on the row
 * step % rows). Two layouts of one program are equivalent when the hashes are
 * equal.
 *
 * Returns false and a reason when the program uses something the evaluator
 * does not model (spills, epochs, FX10 interpolation, registers above R15 as
 * a destination).
 */
bool grate_fp_layout_signature(const struct grate_fp_shader *fp,
                               bool sequential, uint64_t *signature,
                               const char **why);

/**
 * A layout like NVIDIA's (fp/layout2.c): unit operations on any step, registers
 * allocated by row, several slot operations per packet. Replaces the step and
 * packet lists of fp when the symbolic evaluator says the result does what the
 * original (whose sequential signature is seq_signature) did; returns false and
 * leaves fp unchanged otherwise.
 */
bool grate_fp_layout_nvidia(struct grate_fp_shader *fp, uint64_t seq_signature,
                            bool verbose, unsigned old_steps,
                            unsigned old_packets);

/* Copies of the step, packet and MFU lists of fp, and freeing lists. */
bool grate_fp_ir_clone(const struct grate_fp_shader *fp,
                       struct list_head *steps, struct list_head *alus,
                       struct list_head *mfus);
void grate_fp_ir_free(struct list_head *steps, struct list_head *alus,
                      struct list_head *mfus);

#endif
