#ifndef GRATE_PROGRAM_H
#define GRATE_PROGRAM_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include "grate_compiler.h"

struct grate_shader_blob {
   uint32_t *commands;
   int num_commands;
};

struct grate_vertex_shader_state {
   struct pipe_shader_state base;
   struct grate_shader_blob blob;
   uint16_t output_mask;
   uint32_t immediates[GRATE_VP_MAX_IMMEDIATES][4]; /* uploaded with the uniforms, see emit_vs_uniforms() */
   unsigned num_immediates;
};

/* The fragment program with the blend stage for one blend state appended. */
struct grate_blend_variant {
   struct grate_blend_variant *next;
   uint32_t key; /* see grate_fp_blend_key() */
   struct grate_shader_blob blob;
};

struct grate_fragment_shader_state {
   struct pipe_shader_state base;
   struct grate_shader_blob blob;
   unsigned num_fp_instrs;
   unsigned num_alu_instrs;
   struct grate_blend_variant *variants;
   struct grate_fp_info info;

   /* Dual-source blending: the programs of the passes, compiled from the
    * same tokens (see grate_fragment_dual_variant()). */
   struct grate_fp_output output;
   struct grate_fragment_shader_state *dual_variants;
   struct grate_fragment_shader_state *next_dual;
};

struct grate_context;

struct grate_fragment_shader_state *
grate_fragment_dual_variant(struct grate_context *context,
                            struct grate_fragment_shader_state *shader,
                            const struct grate_fp_output *output);

const struct grate_shader_blob *
grate_fragment_blend_blob(struct grate_fragment_shader_state *shader,
                          uint32_t key);

void
grate_context_program_init(struct pipe_context *pcontext);

#endif
