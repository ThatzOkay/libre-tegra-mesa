/*
 * Copyright © 2014-2018 NVIDIA Corporation
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
 */

#include <fcntl.h>

#include "util/os_file.h"
#include "util/u_screen.h"

#include "tegra/tegra_screen.h"
#include "grate/grate_screen.h"
#include "grate/drm/opentegra_lib.h"
#include "tegra_drm_public.h"

static struct pipe_screen *
tegra_or_grate_screen_create(int fd, const struct pipe_screen_config *config,
                             struct renderonly *ro) {
   struct pipe_screen *screen = NULL;
   bool is_grate = false;

#ifdef GALLIUM_GRATE
   is_grate = drm_tegra_get_soc_id() != DRM_TEGRA_UNKNOWN_SOC
           && drm_tegra_get_soc_id() != DRM_TEGRA_INVALID_SOC;

   if (is_grate) {
      screen = grate_screen_create(fd);
   }
#endif

#ifdef GALLIUM_TEGRA
   if (!is_grate) {
      screen = tegra_screen_create(fd);
   }
#endif

   return screen;
}

struct pipe_screen *
tegra_drm_screen_create(int fd, const struct pipe_screen_config *config)
{
   return u_pipe_screen_lookup_or_create(os_dupfd_cloexec(fd), config, NULL,
                                         tegra_or_grate_screen_create);
}
