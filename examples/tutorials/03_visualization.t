-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
import 'ebb'
local L = require 'ebblib'

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'octa.off' )
-- We'll start this program the same way as the last one.

local vdb   = require('ebb.lib.vdb')
-- Then we'll require VDB.  In order to use VDB, you'll need to run
-- an extra installation command, `make vdb`.  See the installation
-- instructions for more details.


local ebb visualize ( v : mesh.vertices )
  vdb.point(v.pos)
end

mesh.vertices:foreach(visualize)
-- Next, we define an Ebb function to plot all of the vertices of the mesh,
-- and execute this function.  (note that VDB will only work while
-- running on the CPU)

-- When we run this program we'll see the output message
--
-- vdb: is the viewer open? Connection refused
--
-- If we want to see the visual output, we need to first start VDB and then
-- run our program.  Once VDB has been installed, the Makefile will create
-- a symlink command `vdb` in the `LISZT_EBB_ROOT` directory.  You can open
-- up a second terminal and run
--
-- ./vdb
--
-- to open up a visualization window, or just launch vdb in the background
--
-- ./vdb &
--
