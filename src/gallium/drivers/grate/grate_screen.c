#include <stdio.h>

#include "compiler/nir/nir_shader_compiler_options.h"

#include "drm-uapi/drm_fourcc.h"

#include "util/u_memory.h"
#include "util/u_screen.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_resource.h"
#include "grate_screen.h"

#include "opentegra_lib.h"

static const struct debug_named_value debug_options[] = {
   { "unimplemented", GRATE_DEBUG_UNIMPLEMENTED,
     "Print unimplemented functions" },
   { "tgsi", GRATE_DEBUG_TGSI,
     "Dump TGSI during program compile" },
   { "trace", GRATE_DEBUG_TRACE,
     "Print trace functions" },
   { NULL }
};

DEBUG_GET_ONCE_FLAGS_OPTION(grate_debug, "GRATE_DEBUG", debug_options, 0)
uint32_t grate_debug;

static void
grate_screen_destroy(struct pipe_screen *pscreen)
{
   grate_trace();
   struct grate_screen *screen = grate_screen(pscreen);

   slab_destroy_parent(&screen->transfer_pool);

   drm_tegra_close(screen->drm);
   FREE(screen);
}

static const char *
grate_screen_get_name(struct pipe_screen *pscreen)
{
   grate_trace();
   return "Tegra";
}

static const char *
grate_screen_get_vendor(struct pipe_screen *pscreen)
{
   grate_trace();
   return "Grate";
}

static const char *
grate_screen_get_device_vendor(struct pipe_screen *pscreen)
{
   grate_trace();
   return "NVIDIA";
}

static int
grate_screen_get_screen_fd(struct pipe_screen *pscreen)
{
   return grate_screen(pscreen)->fd;
}

static void
grate_screen_init_caps(struct grate_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;
   grate_trace();

   u_init_pipe_screen_caps(&screen->base, 1);

   /*
   // NV30 reference values, check and uncomment them
   caps->endianness = PIPE_ENDIAN_LITTLE;
   caps->min_map_buffer_alignment = 64;
   caps->max_viewports = 1;
   caps->max_texture_upload_memory_budget = 8 * 1024 * 1024;
   caps->anisotropic_filter = true;
   caps->occlusion_query = true;
   caps->query_time_elapsed = true;
   caps->query_timestamp = true;
   caps->texture_swizzle = true;
   caps->depth_clip_disable = true;
   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_origin_lower_left = true;
   caps->fs_coord_pixel_center_half_integer = true;
   caps->fs_coord_pixel_center_integer = true;
   caps->tgsi_texcoord = true;
   caps->clear_scissored = true;
   caps->query_memory_info = true;
   caps->vertex_input_alignment = PIPE_VERTEX_INPUT_ALIGNMENT_4BYTE;
   caps->depth_bounds_test = true;
   caps->texture_mirror_clamp =
   caps->texture_mirror_clamp_to_edge =
   caps->primitive_restart =
   caps->primitive_restart_fixed_index = false;
   caps->emulate_nonfixed_primitive_restart = false;
   caps->depth_clip_disable_separate = false;
   caps->max_dual_source_render_targets = 0;
   caps->indep_blend_enable = false;
   caps->indep_blend_func = false;
   caps->max_texture_array_layers = 0;
   caps->shader_stencil_export = false;
   caps->vs_instanceid = false;
   caps->vertex_element_instance_divisor = false;
   caps->max_stream_output_buffers = 0;
   caps->stream_output_pause_resume = false;
   caps->stream_output_interleave_buffers = false;
   caps->min_texture_gather_offset = 0;
   caps->max_texture_gather_offset = 0;
   caps->max_stream_output_separate_components = 0;
   caps->max_stream_output_interleaved_components = 0;
   caps->max_geometry_output_vertices = 0;
   caps->max_geometry_total_output_components = 0;
   caps->max_vertex_streams = 0;
   caps->tgsi_can_compact_constants = false;
   caps->texture_barrier = false;
   caps->seamless_cube_map = false;
   caps->seamless_cube_map_per_texture = false;
   caps->cube_map_array = false;
   caps->fragment_color_clamped = false;
   caps->quads_follow_provoking_vertex_convention = false;
   caps->mixed_colorbuffer_formats = false;
   caps->start_instance = false;
   caps->texture_multisample = false;
   caps->texture_buffer_objects = false;
   caps->texture_buffer_offset_alignment = 0;
   caps->query_pipeline_statistics = false;
   caps->texture_border_color_quirk = false;
   caps->max_texel_buffer_elements = 0;
   caps->vs_layer_viewport = false;
   caps->max_texture_gather_components = 0;
   caps->texture_gather_sm5 = false;
   caps->fake_sw_msaa = false;
   caps->texture_query_lod = false;
   caps->sample_shading = false;
   caps->texture_gather_offsets = false;
   caps->vs_window_space_position = false;
   caps->user_vertex_buffers = false;
   caps->compute = false;
   caps->draw_indirect = false;
   caps->multi_draw_indirect = false;
   caps->multi_draw_indirect_params = false;
   caps->fs_fine_derivative = false;
   caps->conditional_render = false;
   caps->conditional_render_inverted = false;
   caps->sampler_view_target = false;
   caps->clip_halfz = false;
   caps->polygon_offset_clamp = false;
   caps->multisample_z_resolve = false;
   caps->resource_from_user_memory = false;
   caps->device_reset_status_query = false;
   caps->max_shader_patch_varyings = 0;
   caps->texture_float_linear = false;
   caps->texture_half_float_linear = false;
   caps->texture_query_samples = false;
   caps->force_persample_interp = false;
   caps->copy_between_compressed_and_plain_formats = false;
   caps->shareable_shaders = false;
   caps->draw_parameters = false;
   caps->shader_pack_half_float = false;
   caps->fs_position_is_sysval = false;
   caps->fs_face_is_integer_sysval = false;
   caps->shader_buffer_offset_alignment = 0;
   caps->invalidate_buffer = false;
   caps->generate_mipmap = false;
   caps->string_marker = false;
   caps->buffer_sampler_view_rgba_only = false;
   caps->surface_reinterpret_blocks = false;
   caps->compressed_surface_reinterpret_blocks_layered = false;
   caps->query_buffer_object = false;
   caps->framebuffer_no_attachment = false;
   caps->robust_buffer_access_behavior = false;
   caps->cull_distance = false;
   caps->shader_group_vote = false;
   caps->max_window_rectangles = 0;
   caps->viewport_subpixel_bits = 0;
   caps->mixed_color_depth_bits = 0;
   caps->shader_array_components = false;
   caps->native_fence_fd = false;
   caps->legacy_math_rules = false;
   caps->doubles = false;
   caps->int64 = false;
   caps->tgsi_tex_txf_lz = false;
   caps->shader_clock = false;
   caps->polygon_mode_fill_rectangle = false;
   caps->sparse_buffer_page_size = 0;
   caps->shader_ballot = false;
   caps->tes_layer_viewport = false;
   caps->post_depth_coverage = false;
   caps->bindless_texture = false;
   caps->nir_samplers_as_deref = false;
   caps->query_so_overflow = false;
   caps->memobj = false;
   caps->load_constbuf = false;
   caps->tile_raster_order = false;
   caps->max_combined_shader_output_resources = 0;
   caps->framebuffer_msaa_constraints = false;
   caps->signed_vertex_buffer_offset = false;
   caps->context_priority_mask = 0;
   caps->fence_signal = false;
   caps->constbuf0_flags = 0;
   caps->packed_uniforms = false;
   caps->conservative_raster_post_snap_triangles = false;
   caps->conservative_raster_post_snap_points_lines = false;
   caps->conservative_raster_pre_snap_triangles = false;
   caps->conservative_raster_pre_snap_points_lines = false;
   caps->conservative_raster_post_depth_coverage = false;
   caps->max_conservative_raster_subpixel_precision_bias = false;
   caps->programmable_sample_locations = false;
   caps->image_load_formatted = false;
   caps->image_atomic_inc_wrap = false;
   caps->image_store_formatted = false;

   caps->max_gs_invocations = 32;
   caps->max_shader_buffer_size = 1 << 27;
   */
   
   //////////////////////////////////////////////////////////////

   // Bool values
   caps->npot_textures = true; // not really, but mesa requires it for now!`
   caps->blend_equation_separate = true;
   caps->vertex_color_unclamped = true;  // probably irrelevant for GLES2
   caps->vertex_color_clamped = false; // probably irrelevant for GLES2
   caps->mixed_framebuffer_sizes = true;
   caps->buffer_map_persistent_coherent = false; // dunno
   caps->uma = true;
   caps->can_bind_const_buffer_as_vertex = false; // TODO: probably
   caps->allow_mapped_buffers_during_execution = false; // TODO: probably
   
   // well, not quite. but perhaps close enough?
   caps->fragment_shader_texture_lod = true;
   caps->fragment_shader_derivatives = true;
   caps->min_texel_offset = 0;
   caps->max_texel_offset = 0;
   caps->max_render_targets = 8; // ???
   caps->max_texture_2d_size = 2048;
   caps->max_texture_3d_levels = 0;
   caps->max_texture_cube_levels = 16; // ???
   caps->glsl_feature_level =
   caps->glsl_feature_level_compatibility = 120; // no clue
   caps->essl_feature_level = 100; // no clue
   caps->constant_buffer_offset_alignment = 4; // DWORD aligned, can do pure data GATHER 
   caps->texture_transfer_modes = PIPE_TEXTURE_TRANSFER_BLIT;
   caps->vendor_id = 0x10de;
   caps->device_id = 0xFFFFFFFF;
   caps->video_memory = 0;
   caps->max_vertex_attrib_stride = (1 << 24) - 1;
   caps->mixed_color_depth_bits = 1; // probably true ?
   caps->fbfetch = 0; // TODO: supported, but let's enable later
   caps->max_varyings = 16;
   caps->supported_prim_modes_with_restart = 0;
   caps->supported_prim_modes = BITFIELD_BIT(MESA_PRIM_POINTS) |
                                 BITFIELD_BIT(MESA_PRIM_LINES) |
                                 BITFIELD_BIT(MESA_PRIM_LINE_LOOP) |
                                 BITFIELD_BIT(MESA_PRIM_LINE_STRIP) |
                                 BITFIELD_BIT(MESA_PRIM_TRIANGLES) |
                                 BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP) |
                                 BITFIELD_BIT(MESA_PRIM_TRIANGLE_FAN);
   caps->min_line_width = 1.0f; // no clue
   caps->min_line_width_aa = 1.0f; // no clue
   caps->max_line_width = 8192.0; // no clue
   caps->max_line_width_aa = 8192.0; // no clue

   caps->min_point_size = 1.0; // no clue
   caps->min_point_size_aa = 1.0; // no clue
   caps->max_point_size = 8192.0; // no clue
   caps->max_point_size_aa = 8192.0; // no clue
   caps->point_size_granularity = 
   caps->line_width_granularity = 1.0 / 16; // TODO: not a real limit, HW uses floats... but helps caching CSOs
   caps->max_texture_anisotropy = 0; // TODOD: 16.0; // Vendor GL has EXT_texture_filter_anisotropic
   caps->max_texture_lod_bias = 15.0;
   caps->min_conservative_raster_dilate = 0.0;
   caps->max_conservative_raster_dilate = 0.0;
   caps->conservative_raster_dilate_granularity = 0.0;
}

static void
grate_screen_init_shader_caps(struct grate_screen *screen)
{

   struct pipe_shader_caps *caps = NULL;
   grate_trace();

   // Vertex Shader caps
   caps = (struct pipe_shader_caps *)&screen->base.shader_caps[MESA_SHADER_VERTEX];

   // UInt values
   caps->max_instructions = 1024;
   caps->max_alu_instructions = 1024;
   caps->max_tex_instructions = 0;
   caps->max_tex_indirections = 0;
   caps->max_control_flow_depth = 0;
   caps->max_inputs = 16;
   caps->max_outputs = 16;
   caps->max_const_buffer0_size = 256 * sizeof(float[4]);
   caps->max_const_buffers = 1;
   caps->max_temps = 64*4; // 64 vec4s
   caps->max_texture_samplers = 0;
   caps->max_sampler_views = 0;
   caps->max_shader_buffers = 0;
   caps->max_shader_images = 0;
   caps->max_hw_atomic_counters = 0;
   caps->max_hw_atomic_counter_buffers = 0;
   caps->supported_irs = (1 << PIPE_SHADER_IR_TGSI) | (1 << PIPE_SHADER_IR_NIR);

   // Bool values
   caps->cont_supported = false;
   caps->indirect_temp_addr = false; // cannot index attributes, varyings nor GPRs
   caps->indirect_const_addr = true; // can index constant registers 
   caps->subroutines = false;
   caps->integers = false;
   caps->int64_atomics = false;
   caps->fp16 = false;
   caps->fp16_derivatives = false;
   caps->fp16_const_buffers = false;
   caps->int16 = false;
   caps->glsl_16bit_consts = false;
   caps->glsl_16bit_load_dst = false;
   caps->tgsi_sqrt_supported = true;
   caps->tgsi_any_inout_decl_range = false;

   // Fragment Shader caps
   caps = (struct pipe_shader_caps *)&screen->base.shader_caps[MESA_SHADER_FRAGMENT];

   // UInt values
   caps->max_instructions = 4 * 128;
   caps->max_alu_instructions = 4 * 128;
   caps->max_tex_instructions = 128;
   caps->max_tex_indirections = 128;
   caps->max_inputs = 16;
   caps->max_outputs = 16;
   caps->max_const_buffer0_size = 32;
   caps->max_const_buffers = 1;
   caps->max_temps = 16; // scalars
   caps->max_texture_samplers = 16;
   caps->max_sampler_views = 16;
   caps->max_shader_buffers = 0;
   caps->max_shader_images = 0;
   caps->max_hw_atomic_counters = 0;
   caps->max_hw_atomic_counter_buffers = 0;
   caps->supported_irs = (1 << PIPE_SHADER_IR_TGSI) | (1 << PIPE_SHADER_IR_NIR);

   // Bool values
   caps->integers = false;
   caps->int64_atomics = false;
   caps->fp16 = false;
   caps->fp16_derivatives = false;
   caps->fp16_const_buffers = false;
   caps->int16 = false;
   caps->glsl_16bit_consts = false;
   caps->glsl_16bit_load_dst = false;
   caps->tgsi_sqrt_supported = true;
   caps->tgsi_any_inout_decl_range = false;

   /* no control flow */
   caps->max_control_flow_depth = 0;
   caps->cont_supported = false;
   caps->subroutines = false;

   /* no indirection */
   caps->indirect_temp_addr = 0;
   caps->indirect_const_addr = 0;
}


static bool
grate_screen_is_format_supported(struct pipe_screen *pscreen,
                                 enum pipe_format format,
                                 enum pipe_texture_target target,
                                 unsigned sample_count,
                                 unsigned storage_sample_count,
                                 unsigned usage)
{
   if (usage & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DEPTH_STENCIL)) {
      if (grate_pixel_format(format) < 0)
         return false;
   }

   return true;
}

static void
grate_screen_fence_reference(struct pipe_screen *pscreen,
                             struct pipe_fence_handle **ptr,
                             struct pipe_fence_handle *fence)
{
   grate_unimplemented();
}

static bool
grate_screen_fence_finish(struct pipe_screen *screen,
                          struct pipe_context *ctx,
                          struct pipe_fence_handle *fence,
                          uint64_t timeout)
{
   grate_unimplemented();
   return false;
}

static const uint64_t grate_available_modifiers[] = {
   DRM_FORMAT_MOD_LINEAR,
};

static void
grate_screen_query_dmabuf_modifiers(struct pipe_screen *pscreen,
                                    enum pipe_format format, int max,
                                    uint64_t *modifiers,
                                    unsigned int *external_only,
                                    int *count)
{
   int num_modifiers = ARRAY_SIZE(grate_available_modifiers);

   if (!modifiers) {
      *count = num_modifiers;
      return;
   }

   *count = MIN2(max, num_modifiers);
   for (int i = 0; i < *count; i++) {
      modifiers[i] = grate_available_modifiers[i];
      if (external_only)
         external_only[i] = false;
   }
}

static const nir_shader_compiler_options grate_base_compiler_options = {
   /*
   .fuse_ffma32 = true,
   .fuse_ffma64 = true,
   .lower_bitops = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_fdiv = true,
   .lower_fsat = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_fdph = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_fmod = true,
   .lower_fpow = true,
   .lower_uniforms_to_ubo = true,
   .lower_vector_cmp = true,
   .force_indirect_unrolling = nir_var_all,
   .force_indirect_unrolling_sampler = true,
   .max_unroll_iterations = 32,
   */
   .no_integers = true,
};

static bool
grate_screen_is_dmabuf_modifier_supported(struct pipe_screen *pscreen,
                                          uint64_t modifier,
                                          enum pipe_format format,
                                          bool *external_only)
{
   for (int i = 0; i < ARRAY_SIZE(grate_available_modifiers); i++) {
      if (grate_available_modifiers[i] == modifier) {
         if (external_only)
            *external_only = false;

         return true;
      }
   }

   return false;
}

struct pipe_screen *
grate_screen_create(int fd)
{
   struct drm_tegra_channel *drm_channel;
   struct grate_screen *screen;
   int err;

   grate_debug = debug_get_option_grate_debug();
   grate_trace();

   screen = CALLOC_STRUCT(grate_screen);
   if (!screen)
      return NULL;

   screen->fd = fd;
   err = drm_tegra_new(&screen->drm, fd);
   if (err) {
      fprintf(stderr, "drm_tegra_new err: %d\n", err);
      FREE(screen);
      return NULL;
   }

   err = drm_tegra_channel_open(&drm_channel, screen->drm, DRM_TEGRA_GR3D);
   if (err) {
      fprintf(stderr, "drm_tegra_channel_open err: %d\n", err);
      drm_tegra_close(screen->drm);
      FREE(screen);
      return NULL;
   }

   drm_tegra_channel_close(drm_channel);

   screen->base.destroy = grate_screen_destroy;
   screen->base.get_name = grate_screen_get_name;
   screen->base.get_vendor = grate_screen_get_vendor;
   screen->base.get_device_vendor = grate_screen_get_device_vendor;
   screen->base.get_screen_fd = grate_screen_get_screen_fd;
   screen->base.context_create = grate_screen_context_create;
   screen->base.is_format_supported = grate_screen_is_format_supported;
   screen->base.query_dmabuf_modifiers = grate_screen_query_dmabuf_modifiers;
   screen->base.is_dmabuf_modifier_supported = grate_screen_is_dmabuf_modifier_supported;
   
   screen->base.nir_options[MESA_SHADER_VERTEX] = &grate_base_compiler_options;
   screen->base.nir_options[MESA_SHADER_FRAGMENT] = &grate_base_compiler_options;

   /* fence functions */
   screen->base.fence_reference = grate_screen_fence_reference;
   screen->base.fence_finish = grate_screen_fence_finish;

   grate_screen_resource_init(&screen->base);

   grate_screen_init_shader_caps(screen);
   grate_screen_init_caps(screen);

   slab_create_parent(&screen->transfer_pool, sizeof(struct pipe_transfer), 16);

   return &screen->base;
}
