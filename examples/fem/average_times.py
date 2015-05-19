#!/usr/bin/env python

import re
import sys

input_filename = sys.argv[1]
input_file     = open(input_filename)

ignore_first   = True

time_strings = ["Time to assemble force and stiffness", "Time for solver", "Time for step"]
ignore_next  = [ignore_first, ignore_first, ignore_first]
time         = [0.0, 0.0, 0.0]
count        = [0,   0,   0]

for line in input_file:
    for i in range(0,3):
        if time_strings[i] in line:
            if ignore_next[i]:
                ignore_next[i] = False
            else:
                time[i]  = time[i]  + float( re.findall("\d+.\d+", line)[0] )
                count[i] = count[i] + 1
            continue

for i in range(0,3):
    time[i] = time[i]/(count[i] * pow(10, 6))
    print(("%s = %f seconds") % (time_strings[i], time[i]))
