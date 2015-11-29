-- I/O is a tricky problem.  Usually, one needs to support various
-- pre-existing file formats, and load/dump data efficiently or else I/O
-- times can quickly start to dominate the actual simulation computation.
-- Rather than only supporting a small set of standard file formats, Ebb
-- externalizes I/O code using the DLD interface.  This allows for code to
-- support various file formats to be factored out into libraries, whether
-- provided in the standard distribution or by specific users.

-- For example, Loading or Dumping a field of data to/from a CSV file is
-- supported by the standard library `'ebb.io.csv'`.  In this tutorial, we'll
-- write a file loader and dumper for the OFF file format that we've been
-- loading the bunny and octahedral meshes from.  Rather than get into
-- complexities of the standard library's triangle mesh implementation, we'll
-- build an overly-simplified triangle mesh from scratch.

-- Lastly, this tutorial makes more substantial use of Terra.  Rather than
-- try to explain Terra as well, we recommend that programmers interested in
-- building custom file I/O read the Terra documentation themselves.


import 'ebb'
local L = require 'ebblib'

local PN        = require 'ebb.lib.pathname'
local DLD       = require 'ebb.lib.dld'
local vdb       = require 'ebb.lib.vdb'
-- We start with some standard boilerplate

local C         = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]])
-- Using the same Terra mechanism introduced in Tutorial 11, we can import
-- C standard libraries to this Lua environment.  This will give us access to
-- functions for handling filesystem I/O, as well as the standard lib and
-- string comparison.


local infile    = C.fopen(tostring(PN.scriptdir()..'octa.off'), 'r')
if infile == nil then error('Could not open file') end

local terra readOFFheader( f : &C.FILE )
  var fmtstr : int8[4096]
  var n_vertices : int
  var n_triangles : int
  var unused_zero : int
  C.fscanf(f, "%s %d %d %d", fmtstr, &n_vertices, &n_triangles, &unused_zero)
  if C.ferror(f) ~= 0 or
     C.strcmp(fmtstr, "OFF") ~= 0 or
     unused_zero ~= 0
  then
    C.printf("Bad ID\n")
    C.exit(1)
  end
  return n_vertices, n_triangles
end

local retvals = readOFFheader(infile)
local n_vertices, n_triangles = retvals._0, retvals._1
print(n_vertices, n_triangles)
-- Here we use a Terra function to parse out the file header, and unpack the
-- results back into Lua code.  Notice that all of this code is just a Terra
-- translation of C-code that we'd otherwise write to parse an OFF file.


local vertices  = L.NewRelation {
  size = n_vertices,
  name = 'vertices',
}
local triangles = L.NewRelation {
  size = n_triangles,
  name = 'triangles',
}
vertices:NewField('pos', L.vec3d)
triangles:NewField('v', L.vector( vertices, 3 ))
-- We declare two relations for the two kinds of elements, and define the
-- data fields that we plan to read from the OFF file.


local terra errorCheck( f : &C.FILE, s : rawstring )
  if C.ferror(f) ~= 0 then
    C.printf("%s\n", s)
    C.exit(1)
  end
end

local terra strideAlign( stride_bytes : uint8, align_bytes : uint8 )
  if stride_bytes % align_bytes ~= 0 then
    C.printf("Unexpected intra-type alignment from DLD\n")
    C.exit(1)
  end
  return stride_bytes / align_bytes
end

local terra readVertexPositions( dld : &DLD.C_DLD, f : &C.FILE )
  var size    = dld.dim_size[0]
  if dld.dim_stride[0] ~= 1 then
    C.printf("ReadVertex: Unexpected stride from DLD\n")
    C.exit(1)
  end
  var s = strideAlign( dld.type_stride, sizeof(double) )

  var ptr     = [&double](dld.address)
  for k=0,size do
    C.fscanf(f, '%lf %lf %lf', ptr+k*s+0, ptr+k*s+1, ptr+k*s+2)
    errorCheck(f,"ReadVertex: Error reading position data")
  end
end
vertices.pos:Load(readVertexPositions, infile)
-- If a Terra or C function is passed into `field:Load()` as the first
-- argument, then Ebb assumes that the function takes a DLD pointer as the
-- first argument to describe the data layout.  Any further arguments get
-- passed through to the Terra/C function, which is what happens to the file
-- descriptor here.

-- Within the Terra loading function, we can do some sanity checks on the
-- DLD and then pack data into the exposed memory layout in whatever order
-- we want.  To clean up the code, we factor out some subroutines.


local terra readTriangleVertices( dld : &DLD.C_DLD, f : &C.FILE )
  var size    = dld.dim_size[0]
  if dld.dim_stride[0] ~= 1 or
     dld.type_dims[0] ~= 3 or dld.type_dims[1] ~= 1
  then
    C.printf("ReadTri: Unexpected shape from DLD\n")
    C.exit(1)
  end

  var junk : uint

  if dld.base_type == DLD.KEY_8 then
    var ptr     = [&uint8](dld.address)
    var s       = strideAlign( dld.type_stride, sizeof(uint8) )
    for k=0,size do
      C.fscanf(f, '%u %u %u %u', &junk, ptr+k*s+0, ptr+k*s+1, ptr+k*s+2)
      errorCheck(f,"ReadTri: Error reading vertex data")
    end

  elseif dld.base_type == DLD.KEY_16 then
    var ptr     = [&uint16](dld.address)
    var s       = strideAlign( dld.type_stride, sizeof(uint16) )
    for k=0,size do
      C.fscanf(f, '%u %u %u %u', &junk, ptr+k*s+0, ptr+k*s+1, ptr+k*s+2)
      errorCheck(f,"ReadTri: Error reading vertex data")
    end

  else
    C.printf("Unexpected base_type from DLD\n")
    C.exit(1)
  end
end
triangles.v:Load(readTriangleVertices, infile)

C.fclose(infile)
-- When loading key data, we have to account for Ebb's optimized key encoding.
-- Here, we provide for two variations that are sufficient to handle the
-- octahedron and bunny meshes.  If we were loading a substantially larger
-- mesh, we would need to fill out the `KEY_32` case and for extremely large
-- meshes, the `KEY_64` case.



--vertices.pos:print()
--triangles.v:print()

local ebb visualize ( t : triangles )
  var n = L.double(L.id(t) % 256) / 255.0
  var c = { n, L.fmod(n+0.3,1.0), L.fmod(n+0.6,1.0) }
  vdb.color(c)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end

triangles:foreach(visualize)
-- To verify that we've correctly loaded the data, both position and
-- connectivity, we'll run a visualization kernel over the triangles rather
-- than the vertices, and draw the triangles using the vertex position
-- data through the `v` connection we set up.



local ebb translate ( v : vertices )
  v.pos = { v.pos[0], v.pos[1], v.pos[2] + 1.0 }
end

vertices:foreach(translate)
-- To verify that the output file reflects the results of some Ebb code
-- running, let's translate all the vertices in the positive z-direction.



local outfile = C.fopen(tostring(PN.scriptdir()..'sample_out.off'), 'w')
if outfile == nil then error('could not open file to write') end

-- write header
C.fprintf(outfile, "OFF\n")
C.fprintf(outfile, "%d %d %d\n", vertices:Size(), triangles:Size(), 0)
-- We start output by writing out the file header.


local terra writeVertexPositions( dld : &DLD.C_DLD, f : &C.FILE )
  var size    = dld.dim_size[0]

  var ptr     = [&double](dld.address)
  var s       = strideAlign( dld.type_stride, sizeof(double) )
  for k=0,size do
    C.fprintf(f, '%f %f %f\n', ptr[k*s+0], ptr[k*s+1], ptr[k*s+2])
    errorCheck(f,"WriteOff: Error writing position data")
  end
end

vertices.pos:Dump(writeVertexPositions, outfile)
-- The output process is nearly identical, except we use `Dump()` instead of
-- `Load()`.  Note that while both functions expose the underlying data in
-- the same way, they are not interchangable.  Under the covers, Ebb may keep
-- multiple copies of the data.  For instance, if the data field is GPU
-- resident and we call `Dump`, we will get an up-to-date copy of the data
-- on the CPU to write out, but if we write new data to that buffer, Ebb
-- won't move it back to the GPU.


local terra writeTriangleVertices( dld : &DLD.C_DLD, f : &C.FILE )
  var size    = dld.dim_size[0]

  -- write triangle vertices
  if dld.base_type == DLD.KEY_8 then
    var ptr     = [&uint8](dld.address)
    var s       = strideAlign( dld.type_stride, sizeof(uint8) )
    for k=0,size do
      C.fprintf(f, '3 %u %u %u\n', ptr[k*s+0], ptr[k*s+1], ptr[k*s+2])
      errorCheck(f,"WriteOff: Error writing triangles.v 8-bit data")
    end

  elseif dld.base_type == DLD.KEY_16 then
    var ptr     = [&uint16](dld.address)
    var s       = strideAlign( dld.type_stride, sizeof(uint16) )
    for k=0,size do
      C.fprintf(f, '3 %u %u %u\n', ptr[k*s+0], ptr[k*s+1], ptr[k*s+2])
      errorCheck(f,"WriteOff: Error writing triangles.v 16-bit data")
    end

  else
    C.printf("Unexpected base_type from triangles.v DLD\n")
    C.exit(1)
  end
end

triangles.v:Dump(writeTriangleVertices, outfile)

C.fclose(outfile)
-- Similar to before, we need to check for different possible encodings
-- when dumping out the result.  Now, if we take a look at the output
-- translated octagon, we can see that the coordinates have been translated
-- in the z-direction.



