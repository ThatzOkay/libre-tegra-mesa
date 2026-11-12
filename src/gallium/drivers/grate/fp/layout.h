#ifndef GRATE_FP_LAYOUT_H
#define GRATE_FP_LAYOUT_H

#include <stdbool.h>
#include <stdint.h>

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

#endif
