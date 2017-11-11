#include <stdio.h>

#include "util/u_memory.h"

#include "nir/nir_to_tgsi.h"

#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "tgsi/tgsi_transform.h"

#include "host1x01_hardware.h"
#include "grate_common.h"
#include "grate_context.h"
#include "grate_screen.h"
#include "grate_program.h"
#include "grate_compiler.h"
#include "vp/vpir.h"
#include "tgr_3d.xml.h"

struct grate_vs_tgsi_transform_context {
   struct tgsi_transform_context base;
   unsigned next_temp;
};

static void
grate_vs_tgsi_transform_instruction(struct tgsi_transform_context *ctx,
                                    struct tgsi_full_instruction *inst)
{
   struct grate_vs_tgsi_transform_context *gctx =
      (struct grate_vs_tgsi_transform_context *)ctx;

   int const_index = -1;
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      if (inst->Src[i].Register.File != TGSI_FILE_CONSTANT)
         continue;

      if (const_index < 0)
         const_index = inst->Src[i].Register.Index;
      else if (inst->Src[i].Register.Index != const_index) {
         /* We already have a constant-reads in this instruction,
          * add extra instructions in order to read more
          */
         unsigned temp = gctx->next_temp++;
         tgsi_transform_temp_decl(ctx, temp);
         /* insert MOV to temp */
         tgsi_transform_op1_inst(ctx, TGSI_OPCODE_MOV,
                                 /* dst reg */
                                 TGSI_FILE_TEMPORARY, temp,
                                 TGSI_WRITEMASK_XYZW,
                                 /* src reg */
                                 TGSI_FILE_CONSTANT,
                                 inst->Src[i].Register.Index);

         /* rewrite instruction */
         tgsi_transform_src_reg_xyzw(&inst->Src[i], TGSI_FILE_TEMPORARY, temp);
         inst->Src[i].Register.File = TGSI_FILE_TEMPORARY;
         inst->Src[i].Register.Index = temp;
      }
   }

   ctx->emit_instruction(ctx, inst);
}

static struct tgsi_token *
grate_vs_tgsi_transform(const struct tgsi_token *tokens_in)
{
   const unsigned new_len = tgsi_num_tokens(tokens_in) + 100;

   struct tgsi_shader_info info;
   tgsi_scan_shader(tokens_in, &info);

   struct grate_vs_tgsi_transform_context ctx = {};
   ctx.base.transform_instruction = grate_vs_tgsi_transform_instruction;
   ctx.next_temp = info.file_max[TGSI_FILE_TEMPORARY] + 1;

   return tgsi_transform_shader(tokens_in, new_len, &ctx.base);
}


static void *
grate_create_vs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   struct grate_vertex_shader_state *so =
      CALLOC_STRUCT(grate_vertex_shader_state);

   if (!so)
      return NULL;

   so->base = *template;

   if (template->type == PIPE_SHADER_IR_NIR) {
      so->base.tokens = nir_to_tgsi(template->ir.nir,
                                    pcontext->screen);
      so->base.type = PIPE_SHADER_IR_TGSI;
   }


   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(so->base.tokens, 0);
      fprintf(stderr, "\n");
   }

   struct tgsi_token *new_tokens = grate_vs_tgsi_transform(template->tokens);
   if (!new_tokens)
      return NULL;

   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, new_tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_vp_shader vp;
   grate_tgsi_to_vp(&vp, &parser);

   int num_instructions = list_length(&vp.instructions);
   assert(num_instructions < 256);
   int num_commands = 2 + num_instructions * 4;
   uint32_t *commands = MALLOC(num_commands * sizeof(uint32_t));
   if (!commands) {
      FREE(so);
      return NULL;
   }

   commands[0] = host1x_opcode_imm(TGR3D_VP_UPLOAD_INST_ID, 0);
   commands[1] = host1x_opcode_nonincr(TGR3D_VP_UPLOAD_INST,
                                       num_instructions * 4);

   struct vp_instr *last = list_last_entry(&vp.instructions, struct vp_instr, link);
   int offset = 2;
   list_for_each_entry(struct vp_instr, instr, &vp.instructions, link) {
      bool end_of_program = instr == last;
      grate_vp_pack(commands + offset, instr, end_of_program);
      offset += 4;
   }

   so->blob.commands = commands;
   so->blob.num_commands = num_commands;
   so->output_mask = vp.output_mask;

   return so;
}

static void
grate_bind_vs_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->vshader = so;
}

static void
grate_delete_vs_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static void *
grate_create_fs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   struct grate_fragment_shader_state *so =
      CALLOC_STRUCT(grate_fragment_shader_state);

   if (!so)
      return NULL;

   so->base = *template;

   /* TODO: generate code! */

   return so;
}

static void
grate_bind_fs_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->fshader = so;
}

static void
grate_delete_fs_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_program_init(struct pipe_context *pcontext)
{
   pcontext->create_vs_state = grate_create_vs_state;
   pcontext->bind_vs_state = grate_bind_vs_state;
   pcontext->delete_vs_state = grate_delete_vs_state;

   pcontext->create_fs_state = grate_create_fs_state;
   pcontext->bind_fs_state = grate_bind_fs_state;
   pcontext->delete_fs_state = grate_delete_fs_state;
}
