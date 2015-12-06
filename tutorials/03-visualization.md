---
layout: tutorial
title: "03: Visualizing Simulations"
excerpt: "Basic usage of VDB to generate visual output from Ebb programs; We plot the 6 vertices of the octahedron."
---


```
import 'ebb'
local L = require 'ebblib'

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'octa.off' )
```

We'll start this program the same way as the last one.

```
local vdb   = require('ebb.lib.vdb')
```

Then we'll require VDB.  In order to use VDB, you'll need to run an extra installation command, `make vdb`.  See the installation instructions for more details.

```
local ebb visualize ( v : mesh.vertices )
  vdb.point(v.pos)
end

mesh.vertices:foreach(visualize)
```

Next, we define an Ebb function to plot all of the vertices of the mesh, and execute this function.  (note that VDB will only work while running on the CPU)

When we run this program we'll see the output message

```
vdb: is the viewer open? Connection refused
```

If we want to see the visual output, we need to first start VDB and then run our program.  Once VDB has been installed, the Makefile will create a symlink command `vdb` in the `LISZT_EBB_ROOT` directory.  You can open up a second terminal and run

```
./vdb
```

to open up a visualization window, or just launch vdb in the background

```
./vdb &
```

