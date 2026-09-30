#include <errno.h>
#include <stdio.h>

#include "util/u_memory.h"
#include "util/u_upload_mgr.h"

#include "grate_common.h"
#include "grate_device.h"
#include "grate_context.h"
#include "grate_draw.h"
#include "grate_program.h"
#include "grate_resource.h"
#include "grate_screen.h"
#include "grate_state.h"

static int
grate_channel_create(struct grate_context *context,
                     enum drm_tegra_class class,
                     struct grate_channel **channelp)
{
   struct grate_screen *screen = grate_screen(context->base.screen);
   int err;
   struct drm_tegra_channel *drm_channel;
   struct grate_channel *channel;
   grate_trace();

   err = drm_tegra_channel_open(screen->drm, class, &drm_channel);
   if (err < 0) {
      fprintf(stderr, "%s: drm_tegra_channel_open() failed %d\n", __func__, err);
      return err;
   }

   channel = CALLOC_STRUCT(grate_channel);
   if (!channel) {
      err = -ENOMEM;
      goto err_channel;
   }

   channel->context = context;

   err = grate_stream_create(
      screen->drm,
      drm_channel,
      &channel->stream,
      32768
   );
   if (err < 0) {
      FREE(channel);
      goto err_channel;
   }

   *channelp = channel;

   return 0;
   
err_channel:
   drm_tegra_channel_close(drm_channel);
   
   return err;
}

static void
grate_channel_delete(struct grate_channel *channel)
{
   grate_trace();
   grate_stream_destroy(&channel->stream);
   drm_tegra_channel_close(channel->stream.channel);
   FREE(channel);
}

static void
grate_context_destroy(struct pipe_context *pcontext)
{
   struct grate_context *context = grate_context(pcontext);
   grate_trace();

   slab_destroy_child(&context->transfer_pool);

   grate_channel_delete(context->gr3d);
   grate_channel_delete(context->gr2d);
   FREE(context);
}

static void
grate_context_flush(struct pipe_context *pcontext,
                    struct pipe_fence_handle **pfence,
                    enum pipe_flush_flags flags)
{
   //TODO grate_unimplemented();
}

struct pipe_context *
grate_screen_context_create(struct pipe_screen *pscreen,
                            void *priv, unsigned flags)
{
   struct grate_screen *screen = grate_screen(pscreen);
   int err;
   grate_trace();

   struct grate_context *context = CALLOC_STRUCT(grate_context);
   if (!context)
      return NULL;

   context->drm = screen->drm;
   context->base.screen = pscreen;
   context->base.priv = priv;

   err = grate_channel_create(context, DRM_TEGRA_GR2D, &context->gr2d);
   if (err < 0) {
      fprintf(stderr, "grate_channel_create() failed: %d\n", err);
      FREE(context);
      return NULL;
   }

   err = grate_channel_create(context, DRM_TEGRA_GR3D, &context->gr3d);
   if (err < 0) {
      fprintf(stderr, "grate_channel_create() failed: %d\n", err);
      FREE(context);
      return NULL;
   }

   slab_create_child(&context->transfer_pool, &screen->transfer_pool);

   context->base.destroy = grate_context_destroy;
   context->base.flush = grate_context_flush;
   context->base.stream_uploader = u_upload_create_default(&context->base);
   if (!context->base.stream_uploader) {
      FREE(context);
      return NULL;
   }
   context->base.const_uploader = context->base.stream_uploader;

   grate_context_resource_init(&context->base);
   grate_context_state_init(&context->base);
   grate_context_blend_init(&context->base);
   grate_context_sampler_init(&context->base);
   grate_context_rasterizer_init(&context->base);
   grate_context_zsa_init(&context->base);
   grate_context_program_init(&context->base);
   grate_context_vbo_init(&context->base);
   grate_context_draw_init(&context->base);

   return &context->base;
}
