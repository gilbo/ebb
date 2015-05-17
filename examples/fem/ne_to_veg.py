#!/usr/bin/env python

import sys

in_file_path  = sys.argv[1]
node_file_name = ("%s.node" % in_file_path)
ele_file_name  = ("%s.ele" % in_file_path)
vega_file_name  = sys.argv[2]

node_file = open(node_file_name, 'r')
ele_file  = open(ele_file_name,  'r')
vega_file = open(vega_file_name, 'w')

vertices = False
tets = False
num = False
num_vertices = 0
num_tets = 0

vega_file.write("# Generated from %s.*\n" % in_file_path)

vega_file.write("\n*VERTICES\n")
for line in node_file:
    if '#' not in line:
        vega_file.write(line)

vega_file.write("\n*ELEMENTS\nTET\n")
for line in ele_file:
    if '#' not in line:
        vega_file.write(line)

vega_file.write("\n*MATERIAL defaultMaterial\n")
vega_file.write("ENU, 1000, 1250000, 0.4\n")

vega_file.write("\n*REGION\n")
vega_file.write("allElements, defaultMaterial\n")
