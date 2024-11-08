libtegra_noop_drm_shim = shared_library(
  'tegra_noop_drm_shim',
  'tegra_noop.c',
  include_directories: [inc_include, inc_src],
  dependencies: [dep_drm_shim],
  gnu_symbol_visibility : 'hidden',
  install : true,
)
