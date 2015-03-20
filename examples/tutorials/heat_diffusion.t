import "compiler.liszt" -- Every Liszt File should start with this command

-- The first thing we do in most Liszt simulations is to load
-- a geometric domain library.  The standard library comes with a
-- couple of basic domains and code for loading from some simple file
-- formats.  Here we load a Trimesh from an OFF file.
local ioOff = L.require 'domains.ioOff'

-- As we move into more advanced tutorials, we'll learn how to build our
-- own geometric domain libraries or modify existing ones,
-- but for now let's keep it simple.

-- While we're loading in libraries, let's also grab the pathname
-- library, which just provides a couple of convenience functions
-- for manipulating filesystem paths.
local PN = L.require 'lib.pathname'

-- (In case you're wondering, those dots '.' in the middle of the
--  library filenames are specifying subdirectories.  i.e. the last
--  library load said "pull in the pathname library from subdirectory lib".
--  The actual file is LISZT_ROOT/lib/pathname.t on disk.)

-- Since Liszt is built on top of Terra, we can use Terra's
-- capabilities to quickly pull in C code.  The following line
-- loads the standard math library.
local cmath = terralib.includecstring '#include <math.h>'


------------------------------------------------------------------------------

-- Using the pathname library, we can get a path object representing
-- the directory this script file is currently located in.
-- Since we stuck the bunny mesh OFF file here, we should be good to go
local tri_mesh_filename = PN.scriptdir() .. 'bunny.off'

-- Our Trimesh library supports creating a new Trimesh domain by
-- loading in initial data from an OFF file, so we'll go ahead and do that
local bunny = ioOff.LoadTrimesh(tri_mesh_filename)

------------------------------------------------------------------------------

-- Liszt allows us to define two special kinds of variables
-- independently from the geometric domain: Globals and Constants

-- Globals are values that do not vary over the geometric domain,
-- nor are they local to a particular Liszt function.
-- You can think of them as analogous to Globals in C/C++.

-- Notice that we explicitly provide a Liszt type for each global
-- value when we create it.
local timestep = L.Global(L.double, 0.45)

-- Constants are just like Globals, except we can't change their values
-- Of course, this allows the Liszt compiler to handle Constants more
-- efficiently, so make sure to choose appropriately which kind of
-- value you want to use.
local conduction = L.Constant(L.double, 1.0)

------------------------------------------------------------------------------

-- However, most of the data in a Liszt simulation is
-- not stored in Globals and Constants.
-- It occurs in Fields, which are defined over different elements
-- in the geometric domain.

-- Liszt calls these sets of elements in the domain 'Relations'.
-- For instance, the Trimesh domain defines 3 basic relations:
--    * vertices
--    * edges (directed)
--    * triangles

-- You can think of a Relation like a spreadsheet or database table.
-- Each individual element (vertex, triangle, etc.) is a particular
--   row in this spreadsheet.
-- Each column of this spreadsheet is a different Field of the Relation

-- Here, we define a temperature field.  We specify its name
-- and the type of data it stores
bunny.vertices:NewField('temperature', L.double)

-- Then, we initialize this Field using a Lua function.
-- This Lua function takes the vertex id as input and returns
-- a value for the temperature field at that vertex.
local function initial_temperature(vertex_index)
  if vertex_index == 0 then return 3000.0
                       else return 0.0 end
end
-- Then we tell the temperature field to initialize itself
-- using this function we just wrote.
bunny.vertices.temperature:Load(initial_temperature)

-- We also want to keep around a field to store the
-- change in temperature.
bunny.vertices:NewField('d_temperature', L.double):Load(0.0)

-- Notice how we just appended the load to the end of the NewField()
-- function.  Liszt returns the Field object as the result of
-- the NewField call in order to make it more convenient to
-- intialize fields as you define them.

------------------------------------------------------------------------------

-- At this point, we've successfully loaded the mesh into our
-- Geometric domain, defined some useful variables and
-- set up some fields.  Now we're going to define the
-- Liszt functions that will actually compute the simulation timesteps

-- Think of this as the body of a parallel for loop.
-- For instance, in `compute_update` below, we're defining a
-- parallel computation to be performed for each vertex of the mesh.

-- The 'liszt' keyword here tells us that we're defining
-- a Liszt function, with name compute_update and one argument
-- a vertex from bunny.vertices
local liszt compute_update ( v : bunny.vertices )
  -- Here we define some local variables.
  -- If no explicit type is annotated, Liszt tries to infer one
  var sum_t : L.double = 0.0
  var count            = 0

  -- We can loop over the edges leaving this vertex to aggregate data
  for e in v.edges do
    sum_t += e.head.temperature
    count += 1
  end

  -- Finally, let's compute and store the update
  var avg_t  = sum_t / count
  var diff_t = avg_t - v.temperature
  v.d_temperature = timestep * conduction * diff_t
  -- notice that we used our constant and global values above
end

-- Now, let's define a second function that applies the update
local liszt apply_update ( v : bunny.vertices )
  v.temperature += v.d_temperature
end


------------------------------------------------------------------------------


-- WARNING / EXTRA: the following piece of code is for visualizing
-- the results of the heat diffusion using an external visual debugging tool
-- called VDB.  Please see the special VDB tutorial if you want
-- to learn how the following code works.
local vdb  = L.require('lib.vdb')
local cold = L.Constant(L.vec3f,{0.5,0.5,0.5})
local hot  = L.Constant(L.vec3f,{1.0,0.0,0.0})
local liszt debug_tri_draw ( t : bunny.triangles )
  var avg_temp = 0.0
  for i=0,3 do avg_temp += t.v[i].temperature end
  avg_temp = avg_temp / 3.0

  var scale = L.float(cmath.log(1.0 + avg_temp))
  if scale > 1.0 then scale = 1.0f end

  vdb.color((1.0-scale)*cold + scale*hot)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------

-- Ok, Let's recap.  We 
--    1) Loaded the mesh
--    2) Defined data (Globals, Constants, Fields)
--    3) Defined computations (Liszt functions)
-- Now, we're going to tie everything together by running a simulation
-- loop.  This loop is going to run in Lua and sequentially launch
-- data parallel computations over the relations of the domain.

-- Let's just go ahead and run the computation for 300 timesteps
for i = 1,300 do

  -- in order to execute a function, we map it over the appropriate
  -- relation in the domain.  Here, both functions run on the vertices
  bunny.vertices:map(compute_update)
  bunny.vertices:map(apply_update)

  -- EXTRA: VDB (For visualization)
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    bunny.triangles:map(debug_tri_draw)
  vdb.vend()
  -- END EXTRA
end


------------------------------------------------------------------------------

-- For this simple demonstration, we've omitted writing the output
-- out to file.  See the I/O tutorial for more details.

