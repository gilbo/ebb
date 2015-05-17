#!/usr/bin/env python

import sys

vega_file_name  = sys.argv[1]
out_file_path  = sys.argv[2]
ele_file_name  = ("%s.1.ele" % out_file_path)
node_file_name = ("%s.1.node" % out_file_path)

vega_file = open(vega_file_name, 'r')
ele_file  = open(ele_file_name,  'w')
node_file = open(node_file_name, 'w')

vertices = False
tets = False
num = False
num_vertices = 0
num_tets = 0

for line in vega_file:
    if line[0] == '#':
        continue
    elif '*VERTICES' in line:
        num = True
        vertices = True
    elif 'TET' in line:
        tets = True
        num = True
        vertices = False
    elif vertices:
        words = line.strip().split()
        if num:
            node_file.write("%s 3 0 0\n" % words[0])
            num_vertices = int(words[0])
            num = False
        elif num_vertices > 0:
            node_file.write(line)
            num_vertices -= 1
    elif tets:
        words = line.strip().split()
        if num:
            ele_file.write("%s 4 0\n" % words[0])
            num_tets = int(words[0])
            num = False
        elif num_tets > 0:
            ele_file.write(line)
            num_tets -= 1
