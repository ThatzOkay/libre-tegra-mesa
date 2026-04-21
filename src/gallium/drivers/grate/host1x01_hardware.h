/*
 * Copyright (C) 2012-2013 NVIDIA Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *	Arto Merilainen <amerilainen@nvidia.com>
 */

#ifndef HOST1X01_HARDWARE_H_
#define HOST1X01_HARDWARE_H_

#include <linux/types.h>
#include "hw_host1x01_uclass.h"

/* channel registers */
#define HOST1X_CHANNEL_MAP_SIZE_BYTES 16384
#define HOST1X_SYNC_MLOCK_NUM 16

/* sync registers */
#define HOST1X_CHANNEL_SYNC_REG_BASE   0x3000
#define HOST1X_NB_MLOCKS 16

#define BIT(nr)	(1UL << (nr))

static inline uint32_t host1x_class_host_wait_syncpt(uint32_t indx,
						      uint32_t threshold)
{
	return host1x_uclass_wait_syncpt_indx_f(indx) |
		host1x_uclass_wait_syncpt_thresh_f(threshold);
}

static inline uint32_t host1x_class_host_load_syncpt_base(uint32_t indx,
							  uint32_t threshold)
{
	return host1x_uclass_load_syncpt_base_base_indx_f(indx) |
		host1x_uclass_load_syncpt_base_value_f(threshold);
}

static inline uint32_t host1x_class_host_wait_syncpt_base(uint32_t indx,
							  uint32_t base_indx,
							  uint32_t offset)
{
	return host1x_uclass_wait_syncpt_base_indx_f(indx) |
		host1x_uclass_wait_syncpt_base_base_indx_f(base_indx) |
		host1x_uclass_wait_syncpt_base_offset_f(offset);
}

static inline uint32_t host1x_class_host_incr_syncpt_base(uint32_t base_indx,
							  uint32_t offset)
{
	return host1x_uclass_incr_syncpt_base_base_indx_f(base_indx) |
		host1x_uclass_incr_syncpt_base_offset_f(offset);
}

static inline uint32_t host1x_class_host_incr_syncpt(uint32_t cond,
						     uint32_t indx)
{
	return host1x_uclass_incr_syncpt_cond_f(cond) |
		host1x_uclass_incr_syncpt_indx_f(indx);
}

static inline uint32_t host1x_class_host_indoff_reg_write(uint32_t mod_id,
							  uint32_t offset,
							  bool auto_inc)
{
	uint32_t v = host1x_uclass_indoff_indbe_f(0xf) |
		host1x_uclass_indoff_indmodid_f(mod_id) |
		host1x_uclass_indoff_indroffset_f(offset);
	if (auto_inc)
		v |= host1x_uclass_indoff_autoinc_f(1);
	return v;
}

static inline uint32_t host1x_class_host_indoff_reg_read(uint32_t mod_id,
							 uint32_t offset,
							 bool auto_inc)
{
	uint32_t v = host1x_uclass_indoff_indmodid_f(mod_id) |
		host1x_uclass_indoff_indroffset_f(offset) |
		host1x_uclass_indoff_rwn_read_v();
	if (auto_inc)
		v |= host1x_uclass_indoff_autoinc_f(1);
	return v;
}


/* cdma opcodes */
static inline uint32_t host1x_opcode_setclass(uint32_t class_id,
					      uint32_t offset, uint16_t mask)
{
	return (0 << 28) | (offset << 16) | (class_id << 6) | mask;
}

static inline uint32_t host1x_opcode_incr(uint32_t offset, uint16_t count)
{
	return (1 << 28) | (offset << 16) | count;
}

static inline uint32_t host1x_opcode_nonincr(uint32_t offset, uint16_t count)
{
	return (2 << 28) | (offset << 16) | count;
}

static inline uint32_t host1x_opcode_mask(uint32_t offset, uint16_t mask)
{
	return (3 << 28) | (offset << 16) | mask;
}

static inline uint32_t host1x_opcode_imm(uint32_t offset, uint16_t value)
{
	return (4 << 28) | (offset << 16) | value;
}

static inline uint32_t host1x_mask2(uint8_t x, uint8_t y)
{
	return 1 | (1 << (y - x));
}
#endif
