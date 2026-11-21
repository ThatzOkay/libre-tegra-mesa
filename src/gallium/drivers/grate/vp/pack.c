#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "util/macros.h"

#include "vpir.h"

static unsigned
vp_write_mask(unsigned input)
{
   return (((input >> 0) & 1) << 3) |
          (((input >> 1) & 1) << 2) |
          (((input >> 2) & 1) << 1) |
          (((input >> 3) & 1) << 0);
}

static unsigned
vp_swizzle(enum vp_swz swizzle[4])
{
   return (swizzle[0] << 6) |
          (swizzle[1] << 4) |
          (swizzle[2] << 2) |
          (swizzle[3] << 0);
}

static unsigned
vp_src_reg(struct vp_src_operand op)
{
   union {
      struct __attribute__((packed)) {
         unsigned type : 2;    //  0 .. 1
         unsigned index : 6;   //  2 .. 7
         unsigned swizzle : 8; //  8 .. 15
         unsigned negate : 1;  //  16
      };
      unsigned value;
   } u;

   u.type = op.file;
   u.index = op.index;
   u.swizzle = vp_swizzle(op.swizzle);
   u.negate = op.negate;

   return u.value;
}


void
grate_vp_pack(uint32_t *dst, struct vp_instr *instr, bool end_of_program)
{
   /* we can only handle one output register per instruction */
   assert(instr->vec.dst.file != VP_DST_FILE_OUTPUT ||
          instr->scalar.dst.file != VP_DST_FILE_OUTPUT);

   union {
      struct __attribute__((packed)) {
         unsigned end_of_program : 1;                       //   0
         unsigned constant_relative_addressing_enable : 1;  //   1
         unsigned export_write_index : 5;                   //   2 .. 6
         unsigned scalar_rD_index : 6;                      //   7 .. 12
         unsigned vector_op_write_mask : 4;                 //  13 .. 16
         unsigned scalar_op_write_mask : 4;                 //  17 .. 20
         unsigned rC : 17;                                  //  21 .. 37
         unsigned rB : 17;                                  //  38 .. 54
         unsigned rA : 17;                                  //  55 .. 71
         unsigned attribute_fetch_index : 4;                //  72 .. 75
         unsigned uniform_fetch_index : 10;                 //  76 .. 85
         unsigned vector_opcode : 5;                        //  86 .. 90
         unsigned scalar_opcode : 5;                        //  91 .. 95
         unsigned address_register_select : 2;              //  96 .. 97
         unsigned predicate_swizzle : 8;                    //  98 .. 105
         unsigned predicate_lt : 1;                         // 106
         unsigned predicate_eq : 1;                         // 107
         unsigned predicate_gt : 1;                         // 108
         unsigned condition_check : 1;                      // 109
         unsigned condition_set : 1;                        // 110
         unsigned vector_rD_index : 6;                      // 111 .. 116
         unsigned rA_absolute : 1;                          // 117
         unsigned rB_absolute : 1;                          // 118
         unsigned rC_absolute : 1;                          // 119
         unsigned bit120 : 1;                               // 120
         unsigned condition_register_index : 1;             // 121
         unsigned saturate_result : 1;                      // 122
         unsigned attribute_relative_addressing_enable : 1; // 123
         unsigned export_relative_addressing_enable : 1;    // 124
         unsigned condition_flags_write_enable : 1;         // 125
         unsigned export_vector_write_enable : 1;           // 126
         unsigned bit127 : 1;                               // 127
      };

      uint32_t words[4];
   } tmp = {
      .predicate_lt = 1,
      .predicate_eq = 1,
      .predicate_gt = 1,

      .predicate_swizzle = (0 << 6) | (1 << 4) | (2 << 2) | 3,
   };

   /* find the attribute/uniform fetch-values, and zero out the index
    * for these registers.
    */
   int attr_fetch = -1, uniform_fetch = -1;
   bool relative = false;
   for (int i = 0; i < 3; ++i) {
      switch (instr->vec.src[i].file) {
      case VP_SRC_FILE_ATTRIB:
         assert(attr_fetch < 0 ||
                attr_fetch == instr->vec.src[i].index);
         attr_fetch = instr->vec.src[i].index;
         instr->vec.src[i].index = 0;
         break;

      case VP_SRC_FILE_UNIFORM:
         assert(uniform_fetch < 0 ||
                (uniform_fetch == instr->vec.src[i].index &&
                 relative == instr->vec.src[i].relative));
         uniform_fetch = instr->vec.src[i].index;
         relative = instr->vec.src[i].relative;
         instr->vec.src[i].index = 0;
         break;

      default: /* nothing */
         break;
      }
   }

   switch (instr->scalar.src.file) {
   case VP_SRC_FILE_ATTRIB:
      assert(attr_fetch < 0 ||
             attr_fetch == instr->scalar.src.index);
      attr_fetch = instr->scalar.src.index;
      instr->scalar.src.index = 0;
      break;

      case VP_SRC_FILE_UNIFORM:
         assert(uniform_fetch < 0 ||
                (uniform_fetch == instr->scalar.src.index &&
                 relative == instr->scalar.src.relative));
         uniform_fetch = instr->scalar.src.index;
         relative = instr->scalar.src.relative;
         instr->scalar.src.index = 0;
         break;

      default: /* nothing */
         break;

   }

   tmp.attribute_fetch_index = attr_fetch >= 0 ? attr_fetch : 0;
   tmp.uniform_fetch_index = uniform_fetch >= 0 ? uniform_fetch : 0;
   /* the uniform slot is the fetch index plus the address register (A0.x,
    * select 0), as in NVIDIA's compiler (idx-uniform in
    * tools/gpu/grate-vertex-ops-test.c) */
   tmp.constant_relative_addressing_enable = relative;

   tmp.vector_opcode = instr->vec.op;
   switch (instr->vec.dst.file) {
   case VP_DST_FILE_TEMP:
      tmp.vector_rD_index = instr->vec.dst.index;
      tmp.export_write_index = 31;
      break;

   case VP_DST_FILE_OUTPUT:
      tmp.vector_rD_index = 63; // disable register-write
      tmp.export_vector_write_enable = 1;
      tmp.export_write_index = instr->vec.dst.index;
      break;

   case VP_DST_FILE_UNDEF:
      // assert(0);  // TODO: consult NOP
      break;

   default:
      UNREACHABLE("illegal enum vp_dst_file value");
   }
   tmp.vector_op_write_mask = vp_write_mask(instr->vec.dst.write_mask);

   tmp.scalar_opcode = instr->scalar.op;
   switch (instr->scalar.dst.file) {
   case VP_DST_FILE_TEMP:
      tmp.scalar_rD_index = instr->scalar.dst.index;
      tmp.export_write_index = 31;
      break;

   case VP_DST_FILE_OUTPUT:
      tmp.scalar_rD_index = 63; // disable register-write
      tmp.export_vector_write_enable = 0;
      tmp.export_write_index = instr->scalar.dst.index;
      break;

   case VP_DST_FILE_UNDEF:
      // assert(0);  // TODO: consult NOP
      break;

   default:
      UNREACHABLE("illegal enum vp_dst_file value");
   }
   tmp.scalar_op_write_mask = vp_write_mask(instr->scalar.dst.write_mask);

   tmp.rA = vp_src_reg(instr->vec.src[0]);
   tmp.rA_absolute = instr->vec.src[0].absolute;

   tmp.rB = vp_src_reg(instr->vec.src[1]);
   tmp.rB_absolute = instr->vec.src[1].absolute;

   if (instr->vec.src[2].file != VP_SRC_FILE_UNDEF) {
      tmp.rC = vp_src_reg(instr->vec.src[2]);
      tmp.rC_absolute = instr->vec.src[2].absolute;
   } else if (instr->scalar.src.file != VP_SRC_FILE_UNDEF) {
      tmp.rC = vp_src_reg(instr->scalar.src);
      tmp.rC_absolute = instr->scalar.src.absolute;
   }

   tmp.condition_set = instr->cond.set;
   tmp.condition_flags_write_enable = instr->cond.set;
   if (instr->cond.check) {
      tmp.condition_check = 1;
      tmp.predicate_lt = instr->cond.lt;
      tmp.predicate_eq = instr->cond.eq;
      tmp.predicate_gt = instr->cond.gt;
      tmp.predicate_swizzle = vp_swizzle(instr->cond.swizzle);
   }

   if (instr->scalar.op == VP_SCALAR_OP_BRA) {
      /* Like NVIDIA's compiler (tools/gpu/ar20vp-dump.py): the predicate
       * fields choose the condition although condition_check stays 0, the
       * sources are the NOP's attribute 0, and the target instruction index
       * is the swizzle byte of source C. */
      static const enum vp_swz identity[4] = {VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z,
                                              VP_SWZ_W};
      struct vp_src_operand nop = {.file = VP_SRC_FILE_ATTRIB};
      memcpy(nop.swizzle, identity, sizeof(nop.swizzle));
      assert(instr->branch.target < 256);

      tmp.rA = vp_src_reg(nop);
      tmp.rB = vp_src_reg(nop);
      tmp.rC =
         (vp_src_reg(nop) & ~(0xffu << 8)) | (instr->branch.target << 8);
      tmp.predicate_lt = instr->branch.lt;
      tmp.predicate_eq = instr->branch.eq;
      tmp.predicate_gt = instr->branch.gt;
      tmp.predicate_swizzle =
         vp_swizzle((enum vp_swz[4]){instr->branch.lane, instr->branch.lane,
                                     instr->branch.lane, instr->branch.lane});
   }

   tmp.end_of_program = end_of_program;

   /* copy packed instruction into destination */
   for (int i = 0; i < 4; ++i)
      dst[i] = tmp.words[3 - i];
}
