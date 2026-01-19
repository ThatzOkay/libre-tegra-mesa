#include <stdio.h>

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
grate_init_screen_caps(struct grate_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;
   grate_trace();

   u_init_pipe_screen_caps(&screen->base, 1);

   // Bool values
   //caps->graphics;
   caps->npot_textures = true; // not really, but mesa requires it for now!
   //caps->anisotropic_filter;
   //caps->occlusion_query;
   //caps->query_time_elapsed;
   //caps->texture_shadow_map;
   //caps->texture_swizzle;
   //caps->texture_mirror_clamp;
   caps->blend_equation_separate = true;
   //caps->primitive_restart;
   //caps->primitive_restart_fixed_index;
   //caps->indep_blend_enable;
   //caps->indep_blend_func;
   //caps->fs_coord_origin_upper_left;
   //caps->fs_coord_origin_lower_left;
   //caps->fs_coord_pixel_center_half_integer;
   //caps->fs_coord_pixel_center_integer;
   //caps->depth_clip_disable;
   //caps->depth_clip_disable_separate;
   //caps->depth_clamp_enable;
   //caps->shader_stencil_export;
   //caps->vs_instanceid;
   //caps->vertex_element_instance_divisor;
   //caps->fragment_color_clamped;
   //caps->mixed_colorbuffer_formats;
   //caps->seamless_cube_map;
   //caps->seamless_cube_map_per_texture;
   //caps->conditional_render;
   //caps->texture_barrier;
   //caps->stream_output_pause_resume;
   //caps->tgsi_can_compact_constants;
   caps->vertex_color_unclamped = true;  //probably irrelevant for GLES2
   caps->vertex_color_clamped = false; //probably irrelevant for GLES2
   //caps->quads_follow_provoking_vertex_convention;
   //caps->user_vertex_buffers;
   //caps->compute;
   //caps->start_instance;
   //caps->query_timestamp;
   //caps->texture_multisample;
   //caps->cube_map_array;
   //caps->texture_buffer_objects;
   //caps->buffer_sampler_view_rgba_only;
   //caps->tgsi_texcoord;
   //caps->query_pipeline_statistics;
   caps->mixed_framebuffer_sizes = true;
   //caps->vs_layer_viewport;
   //caps->texture_gather_sm5;
   caps->buffer_map_persistent_coherent = false;
   //caps->fake_sw_msaa;
   //caps->texture_query_lod;
   //caps->sample_shading;
   //caps->texture_gather_offsets;
   //caps->vs_window_space_position;
   //caps->draw_indirect;
   //caps->fs_fine_derivative;
   caps->uma = true;
   //caps->conditional_render_inverted;
   //caps->sampler_view_target;
   //caps->clip_halfz;
   //caps->polygon_offset_clamp;
   //caps->multisample_z_resolve;
   //caps->resource_from_user_memory;
   //caps->resource_from_user_memory_compute_only;
   //caps->device_reset_status_query;
   //caps->texture_float_linear;
   //caps->texture_half_float_linear;
   //caps->depth_bounds_test;
   //caps->texture_query_samples;
   //caps->force_persample_interp;
   //caps->shareable_shaders;
   //caps->copy_between_compressed_and_plain_formats;
   //caps->clear_scissored;
   //caps->draw_parameters;
   //caps->shader_pack_half_float;
   //caps->multi_draw_indirect;
   //caps->multi_draw_indirect_params;
   //caps->multi_draw_indirect_partial_stride;
   //caps->fs_position_is_sysval;
   //caps->fs_point_is_sysval;
   //caps->fs_face_is_integer_sysval;
   //caps->invalidate_buffer;
   //caps->generate_mipmap;
   //caps->string_marker;
   //caps->surface_no_compress;
   //caps->surface_reinterpret_blocks;
   //caps->compressed_surface_reinterpret_blocks_layered;
   //caps->query_buffer_object;
   //caps->query_memory_info;
   //caps->framebuffer_no_attachment;
   //caps->robust_buffer_access_behavior;
   //caps->cull_distance;
   //caps->shader_group_vote;
   //caps->shader_array_components;
   //caps->stream_output_interleave_buffers;
   //caps->native_fence_fd;
   //caps->glsl_tess_levels_as_inputs;
   //caps->legacy_math_rules;
   //caps->fp16;
   //caps->doubles;
   //caps->int64;
   //caps->tgsi_tex_txf_lz;
   //caps->shader_clock;
   //caps->shader_realtime_clock;
   //caps->polygon_mode_fill_rectangle;
   //caps->shader_ballot;
   //caps->tes_layer_viewport;
   caps->can_bind_const_buffer_as_vertex = false;
   caps->allow_mapped_buffers_during_execution = false;
   //caps->post_depth_coverage;
   //caps->bindless_texture;
   //caps->nir_samplers_as_deref;
   //caps->query_so_overflow;
   //caps->memobj;
   //caps->load_constbuf;
   //caps->tile_raster_order;
   //caps->signed_vertex_buffer_offset;
   //caps->fence_signal;
   //caps->packed_uniforms;
   //caps->conservative_raster_post_snap_triangles;
   //caps->conservative_raster_post_snap_points_lines;
   //caps->conservative_raster_pre_snap_triangles;
   //caps->conservative_raster_pre_snap_points_lines;
   //caps->conservative_raster_post_depth_coverage;
   //caps->conservative_raster_inner_coverage;
   //caps->programmable_sample_locations;
   //caps->texture_mirror_clamp_to_edge;
   //caps->surface_sample_count;
   //caps->image_atomic_float_add;
   //caps->query_pipeline_statistics_single;
   //caps->dest_surface_srgb_control;
   //caps->compute_grid_info_last_block;
   //caps->compute_shader_derivatives;
   //caps->image_load_formatted;
   //caps->image_store_formatted;
   //caps->throttle;
   //caps->cl_gl_sharing;
   //caps->prefer_compute_for_multimedia;
   //caps->fragment_shader_interlock;
   //caps->fbfetch_coherent;
   //caps->atomic_float_minmax;
   // well, not quite. but perhaps close enough?
   caps->fragment_shader_texture_lod = true;
   caps->fragment_shader_derivatives = true;
   //caps->texture_shadow_lod;
   //caps->shader_samples_identical;
   //caps->image_atomic_inc_wrap;
   //caps->prefer_imm_arrays_as_constbuf;
   //caps->gl_spirv;
   //caps->gl_spirv_variable_pointers;
   //caps->demote_to_helper_invocation;
   //caps->tgsi_tg4_component_in_swizzle;
   //caps->flatshade;
   //caps->alpha_test;
   //caps->two_sided_color;
   //caps->opencl_integer_functions;
   //caps->integer_multiply_32x16;
   //caps->frontend_noop;
   //caps->nir_images_as_deref;
   //caps->packed_stream_output;
   //caps->viewport_transform_lowered;
   //caps->psiz_clamped;
   //caps->viewport_swizzle;
   //caps->system_svm;
   //caps->viewport_mask;
   //caps->alpha_to_coverage_dither_control;
   //caps->map_unsynchronized_thread_safe;
   //caps->blend_equation_advanced;
   //caps->nir_atomics_as_deref;
   //caps->no_clip_on_copy_tex;
   //caps->shader_atomic_int64;
   //caps->device_protected_surface;
   //caps->prefer_real_buffer_in_constbuf0;
   //caps->gl_clamp;
   //caps->texrect;
   //caps->sampler_reduction_minmax;
   //caps->sampler_reduction_minmax_arb;
   //caps->allow_dynamic_vao_fastpath;
   //caps->emulate_nonfixed_primitive_restart;
   //caps->prefer_back_buffer_reuse;
   //caps->draw_vertex_state;
   //caps->prefer_pot_aligned_varyings;
   //caps->sparse_texture_full_array_cube_mipmaps;
   //caps->query_sparse_texture_residency;
   //caps->clamp_sparse_texture_lod;
   //caps->allow_draw_out_of_order;
   //caps->hardware_gl_select;
   //caps->dithering;
   //caps->fbfetch_zs;
   //caps->timeline_semaphore_import;
   //caps->device_protected_context;
   //caps->allow_glthread_buffer_subdata_opt;
   //caps->null_textures;
   //caps->astc_void_extents_need_denorm_flush;
   //caps->validate_all_dirty_states;
   //caps->has_const_bw;
   //caps->performance_monitor;
   //caps->texture_sampler_independent;
   //caps->astc_decode_mode;
   //caps->shader_subgroup_quad_all_stages;
   //caps->call_finalize_nir_in_linker;
   //caps->mesh_shader;
   //caps->representative_fragment_test;

   // SInt values
   caps->accelerated = 1;
   caps->min_texel_offset = 0;
   caps->max_texel_offset = 0;
   //caps->min_texture_gather_offset;
   //caps->max_texture_gather_offset;

   // UInt values
   //caps->max_dual_source_render_targets;
   caps->max_render_targets = 8; //???
   caps->max_texture_2d_size = 2048;
   caps->max_texture_3d_levels = 0;
   caps->max_texture_cube_levels = 16;
   //caps->max_stream_output_buffers;
   //caps->max_texture_array_layers;
   //caps->max_stream_output_separate_components;
   //caps->max_stream_output_interleaved_components;
   caps->glsl_feature_level =
   caps->glsl_feature_level_compatibility = 120; //no clue
   caps->essl_feature_level = 100; //no clue
   caps->constant_buffer_offset_alignment = 4;
   //caps->timer_resolution;
   //caps->min_map_buffer_alignment;
   //caps->texture_buffer_offset_alignment;
   //caps->linear_image_pitch_alignment;
   //caps->linear_image_base_address_alignment;
   caps->texture_transfer_modes = PIPE_TEXTURE_TRANSFER_BLIT;
   //caps->texture_border_color_quirk;
   //caps->max_texel_buffer_elements;
   //caps->max_viewports;
   //caps->max_geometry_output_vertices;
   //caps->max_geometry_total_output_components;
   //caps->max_texture_gather_components;
   //caps->max_vertex_streams;
   caps->vendor_id = 0x10de;
   caps->device_id = 0xFFFFFFFF;
   caps->video_memory = 0;
   caps->max_vertex_attrib_stride = (1 << 24) - 1;
   //caps->max_shader_patch_varyings;
   //caps->shader_buffer_offset_alignment;
   //caps->pci_group;
   //caps->pci_bus;
   //caps->pci_device;
   //caps->pci_function;
   //caps->max_window_rectangles;
   //caps->viewport_subpixel_bits;
   //caps->rasterizer_subpixel_bits;
   caps->mixed_color_depth_bits = true; // ?? wtf
   caps->fbfetch = 0;
   //caps->sparse_buffer_page_size;
   //caps->max_combined_shader_output_resources;
   //caps->framebuffer_msaa_constraints;
   //caps->context_priority_mask;
   //caps->constbuf0_flags;
   //caps->max_conservative_raster_subpixel_precision_bias;
   //caps->max_gs_invocations;
   //caps->max_shader_buffer_size;
   //caps->max_combined_shader_buffers;
   //caps->max_combined_hw_atomic_counters;
   //caps->max_combined_hw_atomic_counter_buffers;
   //caps->max_texture_upload_memory_budget;
   //caps->max_vertex_element_src_offset;
   caps->max_varyings = 16;
   //caps->dmabuf;
   //caps->clip_planes;
   //caps->max_vertex_buffers;
   //caps->gl_begin_end_buffer_size;
   //caps->glsl_zero_init;
   //caps->max_texture_mb;
   caps->supported_prim_modes_with_restart = 0;
   caps->supported_prim_modes = BITFIELD_BIT(MESA_PRIM_POINTS) |
                                 BITFIELD_BIT(MESA_PRIM_LINES) |
                                 BITFIELD_BIT(MESA_PRIM_LINE_LOOP) |
                                 BITFIELD_BIT(MESA_PRIM_LINE_STRIP) |
                                 BITFIELD_BIT(MESA_PRIM_TRIANGLES) |
                                 BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP) |
                                 BITFIELD_BIT(MESA_PRIM_TRIANGLE_FAN);
   //caps->max_sparse_texture_size;
   //caps->max_sparse_3d_texture_size;
   //caps->max_sparse_array_texture_layers;
   //caps->max_constant_buffer_size;
   //caps->query_timestamp_bits;
   //caps->shader_subgroup_size;
   //caps->shader_subgroup_supported_stages;
   //caps->shader_subgroup_supported_features;
   //caps->multiview;
   //caps->max_timeline_semaphore_difference;

   // Float values
   caps->min_line_width = 1.0;
   caps->min_line_width_aa = 1.0;
   caps->max_line_width = 8192.0;
   caps->max_line_width_aa = 8192.0;
   caps->line_width_granularity = 1.0/16;
   caps->min_point_size = 1.0;
   caps->min_point_size_aa = 1.0;
   caps->max_point_size = 8192.0;
   caps->max_point_size_aa = 8192.0;
   caps->point_size_granularity = 1.0/16;
   caps->max_texture_anisotropy = 0.0;
   caps->max_texture_lod_bias = 16.0;
   caps->min_conservative_raster_dilate = 0.0;
   caps->max_conservative_raster_dilate = 0.0;
   caps->conservative_raster_dilate_granularity = 0.0;
}

static void
grate_init_shader_caps(struct grate_screen *screen)
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
   caps->max_const_buffer0_size = 1024;
   caps->max_const_buffers = 1024;
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
   //caps->int64_atomics;
   caps->fp16 = 0;
   caps->fp16_derivatives = 0;
   caps->fp16_const_buffers = 0;
   caps->int16 = 0;
   caps->glsl_16bit_consts = 0;
   //caps->glsl_16bit_load_dst;
   caps->tgsi_sqrt_supported = true;
   caps->tgsi_any_inout_decl_range = false;

   /*  */

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
   caps->max_const_buffers = 32;
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

   /* fence functions */
   screen->base.fence_reference = grate_screen_fence_reference;
   screen->base.fence_finish = grate_screen_fence_finish;

   grate_screen_resource_init(&screen->base);

   grate_init_shader_caps(screen);
   grate_init_screen_caps(screen);

   slab_create_parent(&screen->transfer_pool, sizeof(struct pipe_transfer), 16);

   return &screen->base;
}
