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
-- This time we load the `ioOff` domain library instead of the `grid`
-- domain library.  OFF is a simple file format for triangle meshes,
-- and `ioOff` defines a wrapper around the standard triangle mesh library
-- to load data from OFF files.  Once we have the library required,
-- we use the library function `LoadTrimesh()` to load an octahedron file.

-- In order to make our program more robust to where it gets called from,
-- we use the Ebb-provided `pathname` library.  This library gives us
-- a simple way to introspect on where this particular program file is
-- located on disk (`PN.scriptdir()`) and reference the OFF file as
-- a relative path from the script itself.  Now we can safely invoke
-- this script from anywhere.


print(mesh.vertices:Size())
print(mesh.edges:Size())
print(mesh.triangles:Size())
-- Having loaded the octahedron, let's print out some simple statistics
-- about how many elements of each kind it contains.  If everything is
-- working fine, we should expect to see `6`, `24`, and `8`.
-- (Why 24?  The standard triangle mesh represents directed
--  rather than undirected edges.)


mesh.vertices.pos:Print()
-- When we load in the triangle mesh, the vertices are assigned positions
-- from the file.  Here, we print those out to inspect.  They should be
-- unit distance away from the origin along each axis.  Take a look at
-- the OFF file itself (it's in plaintext) and you'll see that the
-- positions there correspond to the positions printed out.


local ebb translate ( v : mesh.vertices )
  v.pos += {1,0,0}
end

mesh.vertices:foreach(translate)

mesh.vertices.pos:Print()
-- Finally, we can write an Ebb function to translate all of the vertices,
-- execute it, and then print the results.

