# Copyright 2024 Me
# SPDX-License-Identifier: MIT


rnn_src_path = dir_source_root + '/src/gallium/drivers/grate/genxml/rnndb'

rules_file = 'rules-ng.xsd'
freedreno_schema = files(rules_file)

agx_pack = custom_target(
  't30.xml.h',
  input : ['gen_header.py', 't30.xml', freedreno_schema],
  output : 't30.xml.h',
  command : [prog_python, '@INPUT0@', '--rnn', rnn_src_path, '--xml', '@INPUT1@', 'c-defines'],
  capture : true,
  install: true,
  install_dir: 'headers',
  build_always_stale: true,
)

test = custom_target(
  'tegra.xml.h',
  input : ['gen_header.py', 'tegra.xml', freedreno_schema],
  output : 'tegra.xml.h',
  command : [prog_python, '@INPUT0@', '--rnn', rnn_src_path, '--xml', '@INPUT1@', 'c-defines'],
  capture : true,
  install: true,
  install_dir: 'headers',
)

idep_grate_pack = declare_dependency(
  sources : [agx_pack, test],
  dependencies : dep_valgrind,
  include_directories : include_directories('.'),
)


