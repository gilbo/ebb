% Liszt-Ebb: Tutorials

# Getting Started with Liszt-Ebb

## Liszt is...

A project at Stanford University to develop domain-specific languages for physical simulation.  Liszt is focused on _performance portability_.  Performance portable programs are programs that can be run _efficiently_ on a variety of different parallel systems/platforms/architectures. (e.g. CPU, GPU, Multi-core, Clusters, Supercomputers)

Documentation and artifacts for the [original Liszt project/language](http://graphics.stanford.edu/hackliszt) can be found online.

Ebb is the primary DSL for the Liszt project, with specialized DSLs for collision detection and other problems in the works.



## Ebb is...

A Lua-embedded DSL for writing simulations.  Porting an Ebb program from CPU to GPU only requires one command line flag.  We think that's pretty cool.

Ebb was designed with a flexible data model that allows for encoding a range of different _domains_.  As a non-exhaustive list, Ebb supports triangle meshes, grids, tetrahedral meshes, and particles.  Furthermore, domain libraries are _user-authorable_ and can be coupled together in user code.  For example, Ebb seamlessly supports coupling particles to a grid, or coupling the vertices of a mesh to a grid.



# Getting Started (Running Code)

For the rest of this manual, we assume a passing familiarity with the [Lua language](http://www.lua.org/).  Specifically, Ebb is embedded in Lua 5.1, though the Lua language is largely feature-stable today.  You can find a number of good tutorials, manuals and documentation online.

## Installation

See the [README](../README.md) for installation instructions.

## Hello, 42!

Since Ebb doesn't support string values, let's do some arithmetic instead

```
import 'ebb'

local GridLibrary = require 'ebb.domains.grid'

local grid = GridLibrary.NewGrid2d {
  size          = {2,2},
  origin        = {0,0},
  width         = {2,2},
}

local ebb printsum( c : grid.cells )
  L.print(21 + 21)
end

grid.cells:foreach(printsum)
```

Save this in a file `hello42.t`.  Then, execute the command `./ebb hello42.t` to run the Ebb program.  This will print out `42` 4 times, once for each of the 4 grid cells in the 2x2 grid we created.

## Adding Ebb to your Path

If you want to be able to call Ebb from anywhere on your system, you'll want to add the [`LISZT_EBB_ROOT/bin`](../bin) directory to your system path.


## Using Ebb from `C` code

Currently the Ebb interpreter cannot be called from C code as a library, though this is relatively easy to do.  Please contact the developers if this use case is important to you.



# Overview

The tutorials are organized into 3 groups.  The first group (*Intro*) provides a tour through all of the basic features of Ebb, sufficient to get started writing your own simulations using the standard domains.  The second group (*Interop*) introduces the features that let Ebb interoperate with other C-code, including how to write custom high-performance File I/O libraries.  The third and last group (*Domains*) explains the features that are used to write the standard geometric domain libraries; after reading these tutorials, a programmer should be prepared to start developing their own custom geometric domains.

|    | Tutorial                             |  Dependencies       |
| -- | ------------------------------------ | ------------------- |
| *Intro* |                                 |                     |
| 01 | Hello                                |  --                 |
| 02 | Loading                              |  01                 |
| 03 | Visualization                        |  02                 |
| 04 | Fields & Globals                     |  03                 |
| 05 | Neighbor Access                      |  04                 |
| 06 | Phase Checking                       |  05                 |
| 07 | Standard Grid                        |  06                 |
| 08 | Relations                            |  07                 |
| 09 | Particle-Grid Coupling               |  08                 |
| *Interop* |                               |                     |
| 10 | DLDs (Interop)                       |  08                 |
| 11 | Calling C code (Interop)             |  10                 |
| 12 | File I/O (Interop)                   |  11                 |
| *Domains* |                               |                     |
| 13 | Group-By (Domain Libraries)          |  08                 |
| 14 | Join-Tables (Domain Libraries)       |  13                 |
| 15 | Macros (Domain Libraries)            |  14                 |
| 16 | Grid Relations (Domain Libraries)    |  15                 |
| 17 | Subsets (Domain Libraries)           |  16                 |




# Tutorials (Writing Code by Example)

## 01: Hello, 42!

Let's take a look at everything going on in the "Hello, 42!" example program.

```
import 'ebb'
```
The program starts by importing the Ebb language.  Every file you write that includes Ebb code (rather than pure Lua) needs to begin with this command.

```
local GridLibrary = require 'ebb.domains.grid'
```
After importing Ebb, we usually `require` some number of support libraries.  In particular, we'll usually want to require at least one geometric domain library.  Ebb provides a set of default domain libraries available at `'ebb.domains.xxxx'`; Here we use the grid library.

```
local grid = GridLibrary.NewGrid2d {
  size          = {2,2},
  origin        = {0,0},
  width         = {2,2},
}
```
Using the grid library, we can create a new domain.  Here we're telling the library that we want a 2x2 (`size`) 2d grid, with its origin at (0,0) and with grid width 2 in each dimension.

```
local ebb printsum( c : grid.cells )
  L.print(21 + 21)
end
```
After creating a domain, we define computations over that domain using Ebb functions.  Here, we define a function `printsum` which takes a cell of the grid as its argument.  We sometimes call functions that take domain elements as their only argument _kernels_.  These kernels represent data parallel computations over the geometric domain.

```
grid.cells:foreach(printsum)
```
Finally, we invoke this function for each cell in the grid.  Since there are 4 cells, this will print out the sum 4 times.







## 02: Domain Loading From Files

Except in the case of grids, we'll want to load the domain data from a file.  This example program demonstrates loading an octahedron triangle mesh.

```
import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('examples/livecode_getting_started/octa.off')
```
This time we load the `ioOff` domain library instead of the `grid` domain library.  OFF is a simple file format for triangle meshes, and `ioOff` defines a wrapper around the standard triangle mesh library to load data from OFF files.  Once we have the library required, we use the library function `LoadTrimesh()` to load an octahedron file.  Unfortunately, because of how we specify the filepath here, this example can only be executed successfully from `LISZT_EBB_ROOT`.  Instead, let's do something more robust and only slightly more complicated.

```
local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'octa.off' )
```
In this version we load the Ebb-provided `pathname` library to help us manipulate filesystem paths.  Using this library, we can introspect on where this particular program file is located on disk (`PN.scriptdir()`) and reference the OFF file from there.  Now this script can be run safely from anywhere.

```
print(mesh.vertices:Size())
print(mesh.edges:Size())
print(mesh.triangles:Size())
```
Having loaded the octahedron, let's print out some simple statistics about how many elements of each kind it contains.  If everything is working fine, we should expect to see `6`, `24`, and `8`.  (Why 24?  The standard triangle mesh represents directed rather than undirected edges.)

```
mesh.vertices.pos:print()
```
When we load in the triangle mesh, the vertices are assigned positions from the file.  Here, we print those out to inspect.  They should be unit distance away from the origin along each axis.  Take a look at the OFF file itself (it's in plaintext) and you'll see that the positions there correspond to the positions printed out.

```
local ebb translate ( v : mesh.vertices )
  v.pos += {1,0,0}
end

mesh.vertices:foreach(translate)

mesh.vertices.pos:print()
```
Finally, we can write an Ebb function to translate all of the vertices, execute it, and then print the results.








## 03: Visualizing Simulations

```
import "ebb"

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






## 04: User-defined Fields and Globals

We're going to try to make the octahedron oscillate using a sinusoid here.  This won't really require simulating anything yet, but we'll see how to define some data.

```
import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'octa.off' )

local vdb   = require('ebb.lib.vdb')
```
The same start to our program.

```
mesh.vertices:NewField('q', L.vec3d):Load({0,0,0})
```
Because `v.pos` stores the original position of the points, and because we want to remember that information, we're going to have to define a _new field_ to hold modified positions instead.  Here we define that field on the vertices of the triangle mesh.  This defines a `vec3d` quantity (a vector of 3 doubles) for each vertex.  Then, we load in the initial value `{0,0,0}` everywhere to initialize the field.

```
local time = L.Global(L.double, 0)
```
In order to control the oscillation, we're going to define a global time variable.  We give it type `double` and initial value `0`.

```
local ebb set_oscillation ( v : mesh.vertices )
  v.q = 0.5*( L.sin(time) + 1) * v.pos
end

mesh.vertices:foreach(set_oscillation)
```
Finally, we can define the oscillation function.  It will take the original position of each vertex and scale it by a sinusoid-modulated amount, using the time as parameter to the sinusoid.

```
local ebb visualize ( v : mesh.vertices )
  vdb.color({1,1,0})
  vdb.point(v.q)
end
```
And as before, we'll want a simple visualization function.  This time, we'll plot the point coordinates from `v.q` rather than `v.pos` and we'll set all the points' colors to be yellow.

```
for i=1,360 do
  for k=1,40000000 do end

  time:set(i * math.pi / 180.0)
  mesh.vertices:foreach(set_oscillation)

  vdb.vbegin()
  vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()
end
```
Finally, we'll end this file with something that looks a bit more like a real simulation loop.

To start, we loop pointlessly for a long time in order to slow down the loop.  Old videogames did this sometimes.  Hacks with history.

Then, we'll set the global variable using the current loop iteration, and use the set_oscillation function to compute the new vertex positions for the entire mesh.

Finally, we'll wrap our visualization call in a few extra VDB calls to tell VDB to start a frame `vdb.vbegin()`, to clear the screen `vdb.frame()` and finally end the frame `vdb.vend()`.

Suppose you wanted to have the color oscillate together with the positions of the points.  Try modifying the program to do that.









## 05: Accessing Neighbors

In order to do useful simulation computations, we need to not just execute computations at each element of a mesh.  We also need those computations to access data from _neighboring_ elements.  That is, we need some way to access the mesh's _topology_ inside of our functions.

```
import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'bunny.off' )

local vdb   = require('ebb.lib.vdb')
```
We'll start this program again the same way, except we'll load the Stanford bunny mesh instead of the octahedron.

```
local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)
```
Here we use an alternative to global variables to define quantities that will be constant over the course of a simulation.  By defining these quantities as constants, we allow for Ebb to more aggressively optimize their use in code.

```
mesh.vertices:NewField('t', L.double):Load(0)
mesh.vertices:NewField('d_t', L.double):Load(0)
```
We also define a temperature field and change in temperature field.  However, if we just ran the simulation with temperature 0 everywhere, nothing would happen.

```
local function init_temperature(idx)
  if idx == 0 then return 1000 else return 0 end
end
mesh.vertices.t:Load(init_temperature)
```
So instead, we use a Lua function (taking the vertex index in a range `0` to `mesh.vertices:Size()-1`) that returns the temperature that each vertex should be initialized to.  We use this function to place a bunch (1000) of units of temperature on the first vertex.


```
local ebb compute_update ( v : mesh.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.t
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.t
  v.d_t = timestep * conduction * diff_t
end

local ebb apply_update ( v : mesh.vertices )
  v.t += v.d_t
end
```
We use two functions to define a step of heat diffusion.  The first function computes a change in temperature and the second one applies it.  (This may seem strange, but let's try to understand what the functions are doing before we figure out why there are two of them; we'll discuss the reason for using two functions in the next example program.)

The first function `compute_update()` loops over the edges of a given vertex `for e in v.edges do ...`, in order to compute the average temperature of the neighboring vertices.  Then we simply update the current temperature to be a bit more similar to the average neighboring temperature.  This is a pretty standard heat diffusion simulation strategy.

Notice that we were able to loop over the edges of the vertex, and then access the temperature at `e.head.t`.  These are two of the forms of neighbor access provided by the triangle mesh domain library.  They allow us to access other nearby elements starting from the _centered_ element of the function that was passed in as the parameter.

```
local ebb visualize ( v : mesh.vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  vdb.point(v.pos)
end
```
As before, we write a simple visualization function for VDB.  This time, we use the color channel to visualize the heat of a point

```
for i=1,360 do
  mesh.vertices:foreach(compute_update)
  mesh.vertices:foreach(apply_update)

  vdb.vbegin()
  vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()
end
```
The simulation loop is almost identical to last time.  If you run the program with visualization, you should see the heat (in red) spread out over the bunny's back.














## 06: Phases, Reads, Writes, Reductions

In example 05, why did we have to break the heat diffusion step into two functions?  Let's look at some alternate ways we could have written that, and which ways won't work.

```
import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'bunny.off' )

local vdb   = require('ebb.lib.vdb')

local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)

mesh.vertices:NewField('t', L.double):Load(0)
mesh.vertices:NewField('d_t', L.double):Load(0)
mesh.vertices:NewField('new_t', L.double):Load(0)

local function init_temperature(idx)
  if idx == 0 then return 1000 else return 0 end
end
mesh.vertices.t:Load(init_temperature)

local ebb visualize ( v : mesh.vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  vdb.point(v.pos)
end
```
We start the program the same way as before, though we define some extra fields for convenience. 


```
local ebb compute_update_fail ( v : mesh.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.t
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.t
  v.t += timestep * conduction * diff_t
end
```
Here's an obvious way we might try to write the diffusion update.  However, if we try to run the simulation using this function, we'll get a _phase-checking error_.  One of the key features of Ebb is that it only lets programmers write programs that will efficiently parallelize on different architectures.  Phase-checking ensures that any kernel execution---i.e. any time a function is executed for all elements---can be run in parallel.

If we try to both read and write the temperature field at the same time, our program will contain data races.  Specifically, we'll get the error `REDUCE(+) Phase is incompatible with READ Phase`, indicating that we cannot reduce and read the `vertices.t` field at the same time.  This is why we broke the computation into two functions.


```
local ebb compute_update_swap ( v : mesh.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.t
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.t
  v.new_t = v.t + timestep * conduction * diff_t
end
```
Rather than storing an update, we could have written the result into a second buffer.  Then we can swap the two buffers between each simulation step.  This will phase-check safely, but is a different computation for computing the same result.


```
mesh.vertices:NewField('degree', L.double):Load(0)
local ebb compute_degree ( v : mesh.vertices )
  for e in v.edges do
    v.degree += 1
  end
end
mesh.vertices:foreach(compute_degree)

local ebb compute_update_edges ( e : mesh.edges )
  var diff_t    = e.head.t - e.tail.t
  var d_t       = timestep * conduction * (diff_t / e.tail.degree)
  e.tail.new_t += d_t
end
```
Another possibile variation is that we could write the computation as a per-edge, rather than a per-vertex computation, though now we need to copy rather than swap the data


```
local max_diff = L.Global(L.double, 0)
local ebb measure_max_diff( e : mesh.edges )
  var diff_t    = e.head.t - e.tail.t
  max_diff max= L.fabs(diff_t)
end
```
In addition to reductions on fields, we can also reduce values into global variables, which is useful for taking measurements of our simulation.  For instance, here we measure the maximum temperature gradient in the mesh.  We can then periodically display this information to the user so that they can keep track of the simulation even if they don't want to watch a live spatial visualization with VDB.


```
for i=1,360 do
  --mesh.vertices:foreach(compute_update_fail)
  
  -- do one step with the vertex and swap method
  mesh.vertices:foreach(compute_update_swap)
  mesh.vertices:Swap('t', 'new_t')

  -- and one step with the edge swap-and-zero method
  mesh.vertices:Copy { from='t', to='new_t' }
  mesh.edges:foreach(compute_update_edges)
  mesh.vertices:Swap('t', 'new_t')

  vdb.vbegin()
  vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    mesh.edges:foreach(measure_max_diff)
    print( 'iteration #'..tostring(i), 'max gradient: ', max_diff:get() )
  end
end
```
Here we've modified the simulation loop to demonstrate the different ways we can compute the diffusion step.  If you uncomment the first line, you'll get the errors discussed above.  More detailed descriptions of what phase-checking allows can be found in the full Ebb documentation.  In practice, you can assume that phase-checking will fail whenever you have a potential data race.

We've also added a bit of code to manage the measurement and reporting of simulation statistics.  We can `set()` and `get()` the global's values from Lua code here before and after the reduction.





## 07: Using Standard Grids

Part of what makes Ebb unique is the ability to handle simulations over very different domains.  In the previous tutorials, other than the hello world, we used a triangle mesh domain.  In this tutorial, we look at how to make use of the standard grid library.


```
import "ebb"

local GridLib   = require 'ebb.domains.grid'

local vdb       = require 'ebb.lib.vdb'
```
We start the program more or less identically, except we pull in the grid domain library instead of the OFF loader wrapper


```
local N = 40

local grid = GridLib.NewGrid2d {
  size   = {N, N},
  origin = {-N/2, -N/2},
  width  = {N, N},
  periodic_boundary = {true,false},
}
```
Now, instead of loading the domain from a file, we can simply create the grid by specifying a set of named arguments.  For a 2d grid, these are:

  - *size* is the discrete dimensions of the grid; i.e. how many
      cells there are in the x and y directions. (here it is 40x40)
  - *origin* is the spatial position of the (0,0) corner of the grid.
      (here we put the center of the grid at coordinate (0,0))
  - *width* is the spatial dimensions of the grid (which we just
      put in 1-to-1 correspondence with the discrete dimension here)
  - *periodic_boundary* is a list of flags indicating whether the
      given dimension should "wrap-around" or be treated as a hard
      boundary.  (This argument is assumed to be {false,false} if
      not supplied) (Here, we set only the x direction to wrap around)


```
local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)
local max_diff    = L.Global(L.double, 0.0)

grid.cells:NewField('t', L.double):Load(0)
grid.cells:NewField('new_t', L.double):Load(0)

local function init_temperature(x_idx, y_idx)
  if x_idx == 4 and y_idx == 6 then return 1000
                               else return 0 end
end
grid.cells.t:Load(init_temperature)
```
Our definition of simulation quantities is more or less the same as for the triangle mesh.  The one difference to remark on is that we define fields over `grid.cells` instead of `mesh.vertices`.


```
local ebb visualize ( c : grid.cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var p2 = c.center
  vdb.point({ p2[0], p2[1], 0 })
end
```
Unsurprisingly, this means that the visualization code is also similar. However, because cells have spatial extent, it doesn't make sense to simply look up their position.  Instead, we ask for their `c.center` coordinates.  (Warning: center is not actually a field, so if you try to write to it, e.g. `c.center = {3,4}` you'll get an error)


```
local ebb update_temperature ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end
```
Our new temperature update step is similar to the double-buffer strategy from Tutorial 06, but we use the special grid mechanism for neighbor access.  Here, we get the temperature from the four neighbors of a cell in the positive and negative x and y directions.  In general, we can get the cell offset from the current one by (x,y) by writing `c(x,y)` and then access any fields on that cell.


```
local ebb measure_max_diff ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  max_diff  max= L.fabs(diff)
end
```
The function to measure the maximum difference is essentially the same.


```
local ebb update_temp_boundaries ( c : grid.cells )
  var avg : L.double
  if c.yneg_depth > 0 then
    avg = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,0).t )
  elseif c.ypos_depth > 0 then
    avg = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,0).t + c(0,-1).t )
  end
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end
```
However, simply applying `update_temperature` is insufficient to handle the non-periodic boundaries in the y-direction.  Instead, we're going to need to somehow impose a boundary condition on our simulation.

By default, a non-periodic direction marks one final layer of cells on each side of the grid as part of the boundary.  Then we can query how deep we are into a particular boundary using `c.yneg_depth` and `c.ypos_depth`. (Note that like `c.center` these "fields" can't be assigned to)


```
for i=1,360 do
  grid.cells.interior:foreach(update_temperature)
  grid.cells.boundary:foreach(update_temp_boundaries)
  grid.cells:Swap('t', 'new_t')

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    grid.cells.interior:foreach(measure_max_diff)
    print( 'iteration #'..tostring(i), 'max gradient: ', max_diff:get() )
  end
end
```
Our simulation loop is mostly the same, but with one major difference.  Rather than run `update_temperature` for each cell, we only run it for each `interior` cell.  Likewise, we then execute the boundary computation only for each boundary cell.  Though we still visualize all the cells with a single call.  (Note that if we ran `update_temperature` on all of the cells, then we would produce array out of bound errors.)






## 08: Relations

Grids and Triangle Meshes are both useful geometric domains to be able to compute over, but they're hardly exhaustive.  What if you want to compute over a tetrahedral mesh?  A less structured mesh of hexahedral elements?  A polygon mesh with polygons of different numbers of sides? A particle system?

The key idea behind Ebb is that a user can implement new geometric domains as re-usable libraries on top of a common _relational_ data model.  While most simulation programmers don't need to understand this relational model, understanding a little bit about relations can help reinforce the ideas from previous tutorials.

In this tutorial, we'll implement heat diffusion on a torus.  But rather than load a triangle-mesh or use the built in grid library, we'll build our geometric domain from scratch.


```
import "ebb"

local vdb   = require('ebb.lib.vdb')
```
We'll start the program in the usual way


```
local N = 36 -- number of latitudes
local M = 48 -- number of meridians

local vertices  = L.NewRelation {
  name = 'vertices',
  size = N*M,
}
```
We can create a new relation with a library call.  Relations are like spreadsheet tables, where `size` specifies the number of rows.

All of the "sets of elements" we've seen in previous tutorials, including vertices, edges, triangles, and cells are all really just _relations_.  All of our Ebb functions are executed once "for each" row of these relational tables.


```
vertices:NewField('up',     vertices)
vertices:NewField('down',   vertices)
vertices:NewField('left',   vertices)
vertices:NewField('right',  vertices)
```
In order to connect the vertices to themselves, we're going to create fields (i.e. columns of the spreadsheet) that hold _keys_ to a relation rather than primitive values like `L.double`.  These _key-fields_ are the simplest way to connect relations together.  To keep things simple, we don't separately model the concept of edges here, so we've just had these neighbor references direclty refer back to the `vertices` relation.


```
local up_keys     = {}
local down_keys   = {}
local left_keys   = {}
local right_keys  = {}
for i=0,N-1 do
  for j=0,M-1 do
    up_keys[ i*M + j + 1 ]    = ((i-1)%N) * M + (j)
    down_keys[ i*M + j + 1 ]  = ((i+1)%N) * M + (j)
    left_keys[ i*M + j + 1 ]  = (i) * M + ((j-1)%M)
    right_keys[ i*M + j + 1 ] = (i) * M + ((j+1)%M)
  end
end
vertices.up:Load(up_keys)
vertices.down:Load(down_keys)
vertices.left:Load(left_keys)
vertices.right:Load(right_keys)
```
However, we still need to specify what that connectivity is.  Rather than specifying connectivity in the Ebb language, we rely on Lua (or Terra) code to handle data initialization / loading.  Here, we first compute and store the indices for each field of keys (0-based) into a corresponding Lua list (1-based indexing).


```
vertices:NewField('pos',L.vec3d)

local vertex_coordinates = {}
for i=0,N-1 do
  for j=0,M-1 do
    local ang_i = 2 * math.pi * (i/N)
    local ang_j = 2 * math.pi * (j/M)
    local r, z = math.cos(ang_i) + 1.5, math.sin(ang_i)
    local x, y = r*math.cos(ang_j), r*math.sin(ang_j)
    vertex_coordinates[ i*M + j + 1 ] = { x, y, z }
  end
end

vertices.pos:Load(vertex_coordinates)
```
Since we're thinking of these vertices as living on the surface of a torus, we compute and store the position of each vertex.  Just like with the key-fields, this is a Lua computation, so we can't expect Ebb to accelerate or parallelize it.  This is probably ok for some simple examples, but we should remember that this could become a bottleneck as our geometric domain grows.


```
local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)
local max_diff    = L.Global(L.double, 0.0)

local function init_temperature(idx)
  if idx == 0 then return 700 else return 0 end
end
vertices:NewField('t', L.double):Load(init_temperature)
vertices:NewField('new_t', L.double):Load(0)

local ebb visualize ( v : vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  vdb.point(v.pos)
end
```
Unsurprisingly, much of the setup for a heat diffusion is exactly the same as before.


```
local ebb update_temperature ( v : vertices )
  var avg_t   = (v.up.t + v.down.t + v.left.t + v.right.t) / 4.0
  var diff_t  = avg_t - v.t
  v.new_t     = v.t + timestep * conduction * diff_t
end
```
We'll need to modify the update step to use the connectivity relationships we established above.


```
local ebb measure_max_diff( v : vertices )
  var avg_t   = (v.up.t + v.down.t + v.left.t + v.right.t) / 4.0
  var diff_t  = avg_t - v.t
  max_diff max= L.fabs(diff_t)
end
```
And similarly for measuring our progress so far.


```
for i=1,360 do
  vertices:foreach(update_temperature)
  vertices:Swap('t', 'new_t')

  vdb.vbegin()
  vdb.frame()
    vertices:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    vertices:foreach(measure_max_diff)
    print( 'iteration #'..tostring(i), 'max gradient: ', max_diff:get() )
  end
end
```
The simulation loop remains largely the same.

Apart from our use of a visualization library, this heat diffusion program is entirely contained in this one file.  So now the domain libraries are a bit more demystified.  Most importantly, we know that we can think of all the data in Ebb as being arranged in spreadsheet tables, with a row for each element and a column for each field.  This model applies equally well to grid data, although we'll leave the details of how to build grid-like geometric domains to further optional tutorials.  Though even if you're not interested in learning how to build geometric domain libraries, this basic relational model will help explain Ebb's more advanced features.

The remaining tutorials are mostly independent of each other.  Depending on your expected use cases, you may be able to get started with your own code now and come back to the other tutorials when you get more time.



## 09: Particle-Grid Coupling

In this tutorial, we'll look at how to couple two geometric domains together, which is often critical for simulating interacting physical phenomena.

```
import "ebb"

local GridLib   = require 'ebb.domains.grid'

local vdb       = require 'ebb.lib.vdb'

local N = 40

local grid = GridLib.NewGrid2d {
  size   = {N, N},
  origin = {0, 0},
  width  = {N, N},
  periodic_boundary = {true,true},
}
```
We start the grid-based heat diffusion the same way.  For simplicity, we're now using only periodic boundaries and placing the origin at 0,0


```
local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)

grid.cells:NewField('t', L.double):Load(0)
grid.cells:NewField('new_t', L.double):Load(0)

local function init_temperature(x_idx, y_idx)
  if x_idx == 4 and y_idx == 6 then return 1000
                               else return 0 end
end
grid.cells.t:Load(init_temperature)

local ebb update_temperature ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end
```
The code for doing the heat diffusion itself is more or less unchanged, though we've omitted a lot of it for simplicity.


```
local particles = L.NewRelation {
  name = "particles",
  size = N*N,
}
```
To model the particles, we don't need a special domain library.  Simply creating a relation to model the particles will suffice.  Here we create one particle for each cell in the grid.


```
local particle_positions = {}
for yi=0,N-1 do
  for xi=0,N-1 do
    particle_positions[ xi*N + yi + 1 ] = { xi + 0.5,
                                            yi + 0.5 }
  end
end
particles:NewField('pos', L.vec2d):Load(particle_positions)
```
We initialize these particles to sit at the center of each cell. (note that we again have to compensate for the difference between Lua's  1-based indexing and the 0-based indexing in Ebb.)


```
particles:NewField('cell', grid.cells)
```
In order to connect the two geometric domains together, we create a field of keys referencing the cells from the particles.  This connection allows us to encode the concept that a particle is located inside a given cell.


```
grid.locate_in_cells(particles, 'pos', 'cell')
```
The standard grid gives us a special function that we can use to populate the dynamic `cell` connection.  `grid.locate_in_cells()` takes 3 argumetns: 1. a relation with a field of keys to `grid.cells`, 2. the name of a field of that relation holding spatial positions of points/particles and 3. the name of the field of keys pointing into `grid.cells`.  Here, we're calling this function early in order to initialize the values of the `particles.cell` field.


```
local ebb wrap( x : L.double )
  return L.fmod(x + 100*N, N)
end

local ebb advect_particle_position ( p : particles )
  -- estimate heat gradient using a finite difference
  var c   = p.cell
  var dt  = { c(1,0).t - c(-1,0).t, c(0,1).t - c(0,-1).t }
  -- and move the particle downwards along the gradient
  var pos = p.pos - 0.1 * timestep * dt
  -- wrap around the position...
  p.pos = { wrap(pos[0]), wrap(pos[1]) }
end
```
We're going to define the particles' motion using the gradient of the grid's temperature field.  At each timestep, the particles should move a little bit towards the cooler regions.  We do this by first looking up the cell the field is located in and then taking a finite difference approximation of the gradient at that grid cell.

One slight complication is that we need to wrap these coordinates around in order to respect the periodic boundary.  To keep the code clean, we use another ebb function to define that wraparound.


```
local ebb visualize_particles ( p : particles )
  vdb.color({ 1, 1, 0 })
  var p2 = p.pos
  vdb.point({ p2[0], p2[1], 0 })
end
```
Rather than visualize the underlying grid, we plot the particles.


```
for i=1,360 do
  grid.cells:foreach(update_temperature)
  grid.cells:Swap('t', 'new_t')

  particles:foreach(advect_particle_position)
  grid.locate_in_cells(particles, 'pos', 'cell')

  vdb.vbegin()
  vdb.frame()
    particles:foreach(visualize_particles)
  vdb.vend()

  if i % 10 == 0 then print( 'iteration #'..tostring(i) ) end
end
```
Finally, our simulation loop is mostly unchanged, except we add two calls: to update the particle positions and to update the connection to the cells.




## 10: Data Layout Descriptors (DLDs)

In this example, we'll see how to get lower level access to the data that Ebb is storing and computing on.  This data access is provided through metadata objects that we call data layout descriptors (DLDs).  These descriptors communicate information about how data is stored in memory so that Ebb-external code can directly manipulate the data without incurring memory copy overheads.

```
import "ebb"

local GridLib   = require 'ebb.domains.grid'

local DLD       = require 'ebb.lib.dld'
local vdb       = require 'ebb.lib.vdb'
```
In addition to libraries we've already seen, we also require the DLD library.


```
local N = 40

local grid = GridLib.NewGrid2d {
  size   = {N, N},
  origin = {0, 0},
  width  = {N, N},
  periodic_boundary = {true,true},
}

local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)

grid.cells:NewField('t', L.double):Load(0)
grid.cells:NewField('new_t', L.double):Load(0)

local function init_temperature(x_idx, y_idx)
  if x_idx == 4 and y_idx == 6 then return 400
                               else return 0 end
end
grid.cells.t:Load(init_temperature)

local ebb visualize ( c : grid.cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var p2 = c.center
  vdb.point({ p2[0], p2[1], 0 })
end

local ebb update_temperature ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end
```
Similarly to tutorial 09, we simplify the grid-based heat-diffusion code by assuming periodic boundaries.

Below we define a function to shuffle the temperature data in a way that would not be allowed by an Ebb `foreach()` call.  This function is defined using Terra, which is a C-alternative language embedded in Lua.  When using Ebb, Terra is always available to you. We could equally well use a function defined in C, and do exactly that in a later tutorial.

Using DLDs, we can incorporate pre-existing linear solvers, an FFT, or other computations that are (i) legacy code, (ii) specially optimized, or (iii) use computation patterns that Ebb does not support or allow.


```
terra tile_shuffle( dld : &DLD.C_DLD )
  var t_ptr   = [&double](dld.address)
  var xstride = dld.dim_stride[0]
  var ystride = dld.dim_stride[1]

  for y=0,20 do
    for x=0,20 do
      var t1_idx  =      x * xstride   +      y * ystride
      var t2_idx  = (x+20) * xstride   + (y+20) * ystride

      var temp      = t_ptr[t1_idx]
      t_ptr[t1_idx] = t_ptr[t2_idx]
      t_ptr[t2_idx] = temp
    end
  end
end
```
This Terra function reads out the address of the temperature data and the appropriate strides to let it iterate over that data.  Because we're writing this function for the very specific case of a 40x40 2d grid of double values, we can simplify the code tremendously.  The function swaps the top left quadrant with the bottom right.  This function doesn't have any physical analogy, but it's a simple example of a computation that we can't encode in Ebb due to phase-checking restrictions.

While using these external computations imposes less restrictions, we also lose Ebb's ability to automatically parallelize the code.  The `tile_shuffle` function depends on data being CPU-resident, and computes the swap as a purely sequential computation.  If we want to swap in parallel on a GPU, then we would have to write an entirely separate function.


```
for i=1,360 do
  grid.cells:foreach(update_temperature)
  grid.cells:Swap('t', 'new_t')

  if i % 200 == 199 then
    local t_dld = grid.cells.t:GetDLD()

    assert( t_dld.version[1] == 1 and t_dld.version[2] == 0 )
    assert( t_dld.base_type      == DLD.DOUBLE )
    assert( t_dld.location       == DLD.CPU )
    assert( t_dld.type_stride    == 8 )
    assert( t_dld.type_dims[1] == 1 and t_dld.type_dims[2] == 1 )
    assert( t_dld.dim_size[1] == 40 and
            t_dld.dim_size[2] == 40 and
            t_dld.dim_size[3] == 1 )

    tile_shuffle( t_dld:toTerra() )
  end

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then print( 'iteration #'..tostring(i) ) end
end
```
Finally, we modify our simulation loop to swap the tiles on the 200th iteration.  This swap proceeds by requesting a Lua form of the DLD object via `:GetDLD()`, asserting that a number of values are what we expect them to be, and finally calling the tile_shuffle function with a Terra version of the DLD object.




## 11: Calling C-code

In this tutorial, we'll repeat the tile-swapping variant of the heat-diffusion simulation from tutorial 10, but instead of implementing the tile swap computation in Terra, we'll use traditional C-code.

That C-code is in file `11___c_code.c` and `11___c_header.h`.  It needs to be separately compiled into a shared object/library.  We've included some sample compilation commands in a makefile in this directory.

### C Code

```
#include "11___c_header.h"
#include "ebb/lib/dld.h"
```
In order to get access to the struct layout, we include the C-header file description of DLDs from the Ebb standard library.

Notice that this version of the `tile_shuffle()` function also reads the dimensions of the 2d grid, rather than assuming it will be 40x40.

```
void tile_shuffle( void * dld_ptr ) {
  DLD *dld        = (DLD*)(dld_ptr);
  double *t_ptr   = (double*)(dld->address);
  int xstride     = (int)(dld->dim_stride[0]);
  int ystride     = (int)(dld->dim_stride[1]);
  int xdim        = (int)(dld->dim_size[0]);
  int ydim        = (int)(dld->dim_size[0]);
  int halfx       = xdim / 2;
  int halfy       = ydim / 2;

  for (int y=0; y<halfy; y++) {
    for (int x=0; x<halfx; x++) {
      int t1_idx  =         x * xstride   +         y * ystride;
      int t2_idx  = (x+halfx) * xstride   + (y+halfy) * ystride;

      double temp   = t_ptr[t1_idx];
      t_ptr[t1_idx] = t_ptr[t2_idx];
      t_ptr[t2_idx] = temp;
    }
  }
}
```

### Ebb Code

```
import "ebb"

local GridLib   = require 'ebb.domains.grid'

local DLD       = require 'ebb.lib.dld'
local vdb       = require 'ebb.lib.vdb'

local N = 40

local grid = GridLib.NewGrid2d {
  size   = {N, N},
  origin = {0, 0},
  width  = {N, N},
  periodic_boundary = {true,true},
}

local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)

grid.cells:NewField('t', L.double):Load(0)
grid.cells:NewField('new_t', L.double):Load(0)

local function init_temperature(x_idx, y_idx)
  if x_idx == 4 and y_idx == 6 then return 400
                               else return 0 end
end
grid.cells.t:Load(init_temperature)

local ebb visualize ( c : grid.cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var p2 = c.center
  vdb.point({ p2[0], p2[1], 0 })
end

local ebb update_temperature ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end
```
The bulk of this simulation is exactly identical to the previous tutorial 10 on DLDs.


```
local PN    = require 'ebb.lib.pathname'
local here  = tostring(PN.scriptdir():abspath())

terralib.linklibrary(here..'/11___c_obj.so')
local C_Import = terralib.includecstring('#include "11___c_header.h"',
                                         {'-I'..here,
                                          '-I'..here..'/../../include'})
```
Here we use the Terra API calls `linklibrary()` and `includecstring()` to first dynamically link the shared object into the process and then import the header file's interface.  To do so, we need to make sure Terra knows where to look for the shared object and for all the header files being included.  Rather than hard-code fragile paths, we again use the pathname library to dynamically determine the correct filesystem paths based on the location of this script file.

Once these calls have run, the C_Import table will contain a Terra/C function `tile_shuffle()`, that we can call to invoke the C code.  For more information on how this whole process works, take a look at the Terra documentation.


```
for i=1,360 do
  grid.cells:foreach(update_temperature)
  grid.cells:Swap('t', 'new_t')

  if i % 200 == 199 then
    local t_dld = grid.cells.t:GetDLD()

    assert( t_dld.version[1] == 1 and t_dld.version[2] == 0 )
    assert( t_dld.base_type      == DLD.DOUBLE )
    assert( t_dld.location       == DLD.CPU )
    assert( t_dld.type_stride    == 8 )
    assert( t_dld.type_dims[1] == 1 and t_dld.type_dims[2] == 1 )
    assert( t_dld.dim_size[3] == 1 )

    C_Import.tile_shuffle( t_dld:toTerra() )
  end

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then print( 'iteration #'..tostring(i) ) end
end
```
The simulation loop then, is nearly identical to the simulation loop from tutorial 10.






## 12: File I/O

I/O is a tricky problem.  Usually, one needs to support various pre-existing file formats, and load/dump data efficiently or else I/O times can quickly start to dominate the actual simulation computation.  Rather than only supporting a small set of standard file formats, Ebb externalizes I/O code using the DLD interface.  This allows for code to support various file formats to be factored out into libraries, whether provided in the standard distribution or by specific users.

For example, Loading or Dumping a field of data to/from a CSV file is supported by the standard library `'ebb.io.csv'`.  In this tutorial, we'll write a file loader and dumper for the OFF file format that we've been loading the bunny and octahedral meshes from.  Rather than get into complexities of the standard library's triangle mesh implementation, we'll build an overly-simplified triangle mesh from scratch.

Lastly, this tutorial makes more substantial use of Terra.  Rather than try to explain Terra as well, we recommend that programmers interested in building custom file I/O read the Terra documentation themselves.

```
import "ebb"

local PN        = require 'ebb.lib.pathname'
local DLD       = require 'ebb.lib.dld'
local vdb       = require 'ebb.lib.vdb'
```
We start with standard library includes


```
local C         = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]])
```
Using the same Terra mechanism introduced in Tutorial 11, we can import C standard libraries to this Lua environment.  This will give us access to functions for handling filesystem I/O, as well as the standard lib and string comparison.


```
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
```
Here we use a Terra function to parse out the file header, and unpack the results back into Lua code.  Notice that all of this code is just a Terra translation of C-code that we'd otherwise write to parse an OFF file.


```
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
```
We declare two relations for the two kinds of elements, and define the data fields that we plan to read from the OFF file.


```
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
```
If a Terra or C function is passed into `field:Load()` as the first argument, then Ebb assumes that the function takes a DLD pointer as the first argument to describe the data layout.  Any further arguments get passed through to the Terra/C function, which is what happens to the file descriptor here.

Within the Terra loading function, we can do some sanity checks on the DLD and then pack data into the exposed memory layout in whatever order we want.  To clean up the code, we factor out some subroutines.


```
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
```
When loading key data, we have to account for Ebb's optimized key encoding.  Here, we provide for two variations that are sufficient to handle the octahedron and bunny meshes.  If we were loading a substantially larger mesh, we would need to fill out the `KEY_32` case and for extremely large meshes, the `KEY_64` case.


```
--vertices.pos:print()
--triangles.v:print()

local ebb visualize ( t : triangles )
  var n = L.double(L.id(t) % 256) / 255.0
  var c = { n, L.fmod(n+0.3,1.0), L.fmod(n+0.6,1.0) }
  vdb.color(c)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end

triangles:foreach(visualize)
```
To verify that we've correctly loaded the data, both position and connectivity, we'll run a visualization kernel over the triangles rather than the vertices, and draw the triangles using the vertex position data through the `v` connection we set up.


```
local ebb translate ( v : vertices )
  v.pos = { v.pos[0], v.pos[1], v.pos[2] + 1.0 }
end

vertices:foreach(translate)
```
To verify that the output file reflects the results of some Ebb code running, let's translate all the vertices in the positive z-direction.


```
local outfile = C.fopen(tostring(PN.scriptdir()..'sample_out.off'), 'w')
if outfile == nil then error('could not open file to write') end

-- write header
C.fprintf(outfile, "OFF\n")
C.fprintf(outfile, "%d %d %d\n", vertices:Size(), triangles:Size(), 0)
```
We start output by writing out the file header.


```
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
```
The output process is nearly identical, except we use `Dump()` instead of `Load()`.  Note that while both functions expose the underlying data in the same way, they are not interchangable.  Under the covers, Ebb may keep multiple copies of the data.  For instance, if the data field is GPU resident and we call `Dump`, we will get an up-to-date copy of the data on the CPU to write out, but if we write new data to that buffer, Ebb won't move it back to the GPU.


```
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
```
Similar to before, we need to check for different possible encodings when dumping out the result.  Now, if we take a look at the output translated octagon, we can see that the coordinates have been translated in the z-direction.





## 13: Group-By and Query-Loops

In tutorial 08 (relations) we saw the most basic mechanisms for building custom geometric domain libraries: the creation of new relations and using fields of keys (_key fields_) to connect those relations together.  Tutorials 13-17 introduce the remaining mechanisms and tricks used to model geometric domains.

In this tutorial, we'll explore the _group-by_ operation.  To do so, we'll use the earlier heat diffusion example from tutorial 8, except we'll use two relations this time: one for the vertices, and one for the edges.  Then, since the edges are explicitly represented, we'll simply omit the edges that form the torus.  Because we explicitly represent the edges in this way, we no longer need to assume that each vertex has exactly four neighbors in each of 4 cardinal directions.

```
import "ebb"

local vdb   = require('ebb.lib.vdb')
```
We start the program as usual.


```
local N = 40

local vertices  = L.NewRelation {
  name = 'vertices',
  size = N*N,
}
```
We create N^2 vertices in the domain.  Unlike in tutorial 08, where we encoded a toroidal topology, we'll just omit the wrap-around edges here.


```
local edges     = L.NewRelation {
  name = 'edges',
  size = 4*N*(N-1),
}
```
And we create 2\*N\*(N-1) horizontal edges, as well as the same number of vertical edges.  These are directed edges.


```
edges:NewField('head', vertices)
edges:NewField('tail', vertices)
```
Each edge needs to identify its head and tail vertex.


```
local tail_keys = {}
local head_keys = {}
local ei        = 1
for i=0,N-1 do
  for j=0,N-1 do
    local vidx = i*N + j
    -- left, right, bottom, top
    if i > 0 then
      tail_keys[ei] = vidx
      head_keys[ei] = (i-1)*N + j
      ei = ei + 1
    end
    if i < N-1 then
      tail_keys[ei] = vidx
      head_keys[ei] = (i+1)*N + j
      ei = ei + 1
    end
    if j > 0 then
      tail_keys[ei] = vidx
      head_keys[ei] = i*N + j-1
      ei = ei + 1
    end
    if j < N-1 then
      tail_keys[ei] = vidx
      head_keys[ei] = i*N + j+1
      ei = ei + 1
    end
  end
end
edges.head:Load(head_keys)
edges.tail:Load(tail_keys)
```
We compute and load the connectivity data for edges using Lua lists.


```
edges:GroupBy('tail')
```
We _group_ the `edges` relation by its `tail` field.  This is a setup operation which tells Ebb how we plan to use the data.  In particular, `GroupBy()` tells Ebb that we plan to "query" / access the edges according to which vertex their tail is.

Another way we can think of the group-by relationship is that it _inverts_ the forward relationship established by the `tail` key-field.  If we think of `tail` as a function from edges to vertices, then group-by allows us to access the pre-image of any vertex: a set of edges pointing to that vertex.  We'll see how this is used inside an Ebb function below.


```
vertices:NewField('pos',L.vec2d)

local vertex_coordinates = {}
for i=0,N-1 do
  for j=0,N-1 do
    vertex_coordinates[ i*N + j + 1 ] = { i, j }
  end
end

vertices.pos:Load(vertex_coordinates)
```
Since the vertices are no longer connected in a toroidal topology, we'll go ahead and give them positions in a grid.


```
local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)
local max_diff    = L.Global(L.double, 0.0)

local function init_temperature(idx)
  if idx == 0 then return 1000 else return 0 end
end
vertices:NewField('t', L.double):Load(init_temperature)
vertices:NewField('new_t', L.double):Load(0)

local ebb visualize ( v : vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  var pos = { v.pos[0], v.pos[1], 0.0 }
  vdb.point(pos)
end

local ebb measure_max_diff( e : edges )
  var diff_t    = e.head.t - e.tail.t
  max_diff max= L.fabs(diff_t)
end
```
Most of the simulation code is the same as before


```
local ebb update_temperature ( v : vertices )
  var sum_t = 0.0
  var count = 0.0

  for e in L.Where(edges.tail, v) do
    sum_t += e.head.t
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.t
  v.new_t = v.t + timestep * conduction * diff_t
end
```
However, the `update_temperature()` function now uses an unfamiliar loop.  In particular, the `L.Where(edges.tail, v)` expression is called a _query_, and the whole loop construct is called a _query loop_.  Read in english, it says "for each `e` in `edges` where `e.tail == v` do ...".  Query loops can only be executed if the target table (`edges` here) has been prepared with a `GroupBy()` operation.  Otherwise, the typechecker will throw an error.


```
for i=1,360 do
  vertices:foreach(update_temperature)
  vertices:Swap('t', 'new_t')

  vdb.vbegin()
  vdb.frame()
    vertices:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    edges:foreach(measure_max_diff)
    print( 'iteration #'..tostring(i), 'max gradient: ', max_diff:get() )
  end
end
```
The simulation loop is unchanged.





## 14: Join Tables

Somewhat surprisingly, key-fields together with group-by/query-loops are sufficient to express arbitrary graph connectivity patterns.  To do so, we use a well known trick from databases, called a _join table_.  Unlike the relational tables we've been declaring up to this point, join tables don't represent a particular set of objects.  Instead, they represent a relationship between two different other sets of objects.  As an example, we'll load in a standard triangle-mesh and augment it with a way to get all of the triangles touching a given vertex.

```
import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'bunny.off' )

local vdb   = require('ebb.lib.vdb')
```
Our program starts in the usual way.


```
local v_triples       = mesh.triangles.v:Dump({})
local tri_ids, v_ids  = {}, {}
for k=0,mesh.triangles:Size()-1 do
  tri_ids[ 3*k + 1 ] = k
  tri_ids[ 3*k + 2 ] = k
  tri_ids[ 3*k + 3 ] = k
  local triple = v_triples[k+1]
  v_ids[ 3*k + 1 ] = triple[1]
  v_ids[ 3*k + 2 ] = triple[2]
  v_ids[ 3*k + 3 ] = triple[3]
end

local triangles_of_vertex = L.NewRelation {
  name = "triangles_of_vertex",
  size = #tri_ids,
}
triangles_of_vertex:NewField('tri', mesh.triangles):Load(tri_ids)
triangles_of_vertex:NewField('v', mesh.vertices):Load(v_ids)
```
This is the join-table.  It contains one row for each triangle-vertex pair that are in contact.  This table now explicitly represents the connection between the triangles and vertices.


```
triangles_of_vertex:GroupBy('v')
```
When we group this join-table by the vertices, we prepare it so that we can quickly query for all the rows with a given vertex.  This will allow us to iterate over all the triangles attached to a given vertex.  If we want to also access this table by the vertices, we'll have to make a second copy that we can group a second way.

Rather than simulate, we're going to visualize the dual-area around the vertices.


```
mesh.vertices:NewField('dual_area', L.double):Load(0.0)
mesh.triangles:NewField('area', L.double):Load(0.0)

local ebb compute_area ( t : mesh.triangles )
  var e01 = t.v[1].pos - t.v[0].pos
  var e02 = t.v[2].pos - t.v[0].pos

  t.area = L.length( L.cross(e01, e02) )
end
mesh.triangles:foreach(compute_area)
```
We compute triangle areas the standard way.


```
local ebb compute_dual_area ( v : mesh.vertices )
  for t in L.Where(triangles_of_vertex.v, v).tri do
    v.dual_area += t.area
  end
  v.dual_area = v.dual_area / 3.0
end
mesh.vertices:foreach(compute_dual_area)
```
Dual areas are computed from the vertices using the triangles_of_vertex join-table we set up.  This is a query loop like we saw in the last tutorial, but with a slight modification.  After the `L.Where(...)` we have a post-fix `.tri` as if we were accessing a field.  In order to simplify the use of join-tables, Ebb allows for this special bit of syntax sugar.


```
local ebb visualize ( v : mesh.vertices )
  var a = L.fmin( L.fmax( v.dual_area * 2.0 - 0.5, 0.0 ), 1.0 )
  vdb.color({ 0.5-a, 0.5 * a + 0.5, 0.5 * a + 0.5 })
  vdb.point(v.pos)
end
mesh.vertices:foreach(visualize)
```
Finally, we visualize the vertex area using a color encoding.







## 15: Macros

In the last two tutorials, we saw a useful feature and trick which allows us to model arbitrary connectivity patterns between relations.  However, these mechanisms require programmers to use an unintuitive interface: `L.Where(...)` loops.  In earlier tutorials, we saw a much more intuitive loop syntax for triangle meshes: `v.edges`.  The missing ingredient is a macro.  In this tutorial, we explain how geometric domain authors can encapsulate and hide relational details behind more intuitive syntax.

Besides macros, we'll also introduce field-functions.  These two tools allow simulation and geometric domain authors to abstract functionality and retrofit old code.

Unlike previous tutorials, this file will not compute much, though it can still be safely executed.

```
import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'octa.off' )

local vdb   = require('ebb.lib.vdb')

local v_triples       = mesh.triangles.v:Dump({})
local tri_ids, v_ids  = {}, {}
for k=0,mesh.triangles:Size()-1 do
  tri_ids[ 3*k + 1 ] = k
  tri_ids[ 3*k + 2 ] = k
  tri_ids[ 3*k + 3 ] = k
  local triple = v_triples[k+1]
  v_ids[ 3*k + 1 ] = triple[1]
  v_ids[ 3*k + 2 ] = triple[2]
  v_ids[ 3*k + 3 ] = triple[3]
end

local triangles_of_vertex = L.NewRelation {
  name = "triangles_of_vertex",
  size = #tri_ids,
}
triangles_of_vertex:NewField('tri', mesh.triangles):Load(tri_ids)
triangles_of_vertex:NewField('v', mesh.vertices):Load(v_ids)

triangles_of_vertex:GroupBy('v')
```
The program starts the same way as in the last tutorial; by defining a join-table.


```
local swap_macro = L.Macro(function(a, b)
  return ebb quote
    var tmp = a
    a = b
    b = tmp
  in 0 end
end)
local ebb use_swap( v : mesh.vertices )
  var a : L.int = 1
  var b : L.int = 2
  swap_macro(a, b)
  L.print(b)
  L.assert(b == 1)
end
mesh.vertices:foreach(use_swap)
```
To start, we define a macro that swaps two values.  This macro is defined by a Lua function that runs at compile time, returning a quoted piece of Ebb code.  This quoted bit of code gets spliced into the ebb function below where swap_macro is called.  That is, the macro gets substituted, rather than executed like a function.  The design here is very similar to Terra, though less fully developed. (Note that the `in 0` is needed in case an Ebb quote is used somewhere where an expression is expected.)

Why not just define swap with another Ebb function?  If we did that, then the two arguments to swap would be passed by value.  Swapping them would accomplish nothing in the calling context.  However, because a macro is substituted, the parameters are really just other bits of code containing the variable symbols/names.  In general, macros are needed in some weird cases like these where we want to break the rules of normal function calls.


```
local triangle_macro = L.Macro(function(v)
  return ebb `L.Where(triangles_of_vertex.v, v).tri
end)
mesh.vertices:NewFieldMacro('triangles', triangle_macro)
```
One of the special features of Ebb is the ability to install macros on relations as if they were fields.  Now that we've installed this macro, we can clean up the code for computing the dual-area of a vertex.


```
mesh.vertices:NewField('dual_area', L.double):Load(0.0)
mesh.triangles:NewField('area', L.double):Load(0.0)

local ebb compute_area ( t : mesh.triangles )
  var e01 = t.v[1].pos - t.v[0].pos
  var e02 = t.v[2].pos - t.v[0].pos

  t.area = L.length( L.cross(e01, e02) )
end
mesh.triangles:foreach(compute_area)

local ebb compute_dual_area ( v : mesh.vertices )
  for t in v.triangles do
    v.dual_area += t.area
  end
  v.dual_area = v.dual_area / 3.0
end
mesh.vertices:foreach(compute_dual_area)
```
Notice that the query loop in `compute_dual_area()` now reads `for t in v.triangles do` rather than `for t in L.Where(...).tri do`.  Even though `L.Where(...)` is not a value that could be returned from a function, we can use a macro to abstract the snippet of code.  By further using the `NewFieldMacro()` feature, we can make the user-syntax clean and uniform.  This is how `v.edges` is defined in the standard triangle library.


```
mesh.vertices:NewField('density', L.double):Load(1.0)
mesh.vertices:NewFieldReadFunction('mass', ebb ( v )
  return v.dual_area * v.density
end)
```
Besides Field Macros, we can also install functions as if they were fields.  This gives us a way to define derived quantities without having to compute and store a new field.  For instance, here mass can be defined in terms of area and density.  If the area changes, then so does the mass.  Field-functions convert to function calls unlike field-macros, which get replaced by a macro-substitution.

When possible, try to use a field function before you resort to a macro. You will generally have an easier time debugging your code and avoiding gotchas.











## 16: Grid Relations

In this tutorial, we show how to create grid-structured domains and connect them.  Grid-structured data is especially important for simulations.  Grids also offer special opportunities for optimization by exploiting the regular addressing to eliminate memory accesses.  Consequently, Ebb provides special mechanisms for indicating that certain relations are grid-structured, and for connecting those relations to themselves and other gridded relations.

To illustrate these features, we're going to write two coupled simulations on grids of different scales.  The lower-resolution grid will simulate a heat diffusion, while the higher-resolution grid will simulate the wave equation.  The particular simulation isn't physically derived, but will show both how to write code for grids and how to construct multi-grid structures.

```
import "ebb"

local vdb   = require('ebb.lib.vdb')
```
We'll start the program in the usual way.


```
local N = 50

local hi_cells = L.NewRelation {
  name      = 'hi_cells',
  dims      = { N, N },
  periodic  = { true, true },
}
local lo_cells = L.NewRelation {
  name      = 'lo_cells',
  dims      = { N/2, N/2 },
  periodic  = { true, true },
}
```
Instead of declaring the size of a relation, we can specify `dims`, a Lua list of 2 or 3 numbers, giving the number of grid entries we want in each dimension.  If we want the grid relation to be considered periodic, then we can additionally specify a `periodic` parameter.  Notice that raw grid structured relations do not need an `origin` or `width` specified.  Those are parameters of the standard library grid, which provides a set of standard functionality on top of the raw grid relations.


```
hi_cells:NewField('t', L.double):Load(function(xi,yi)
  if xi == 4 and yi == 10 then return 400 else return 0 end
end)
lo_cells:NewField('t', L.double):Load(0)
hi_cells:NewField('t_prev', L.double):Load(hi_cells.t)
lo_cells:NewField('t_prev', L.double):Load(lo_cells.t)
hi_cells:NewField('t_next', L.double):Load(hi_cells.t)
lo_cells:NewField('t_next', L.double):Load(lo_cells.t)
```
Here we define the necessary simulation variables.  Rather than explicitly encode velocity, we choose to instead store the previous field value.  By copying, we effectively choose to initialize everything with 0 velocity.


```
local ebb shift_right_example( c : hi_cells )
  var left_c = L.Affine(hi_cells, {{1,0, -1},
                                   {0,1,  0}}, c)
  c.t_next = left_c.t
end
hi_cells:foreach(shift_right_example)
```
This computation doesn't accomplish anything for our simulation, but it does demonstrate how we can use the special `L.Affine(...)` function to access neighboring elements in a grid.  The first argument to `L.Affine()` specifies which grid-structured relation we're performing a lookup into.  The second argument specifies an _affine transformation_ of the third argument's "coordinates." (since the third argument is an key from some grid strucured relation)  This second argument must be a constant matrix, which we can interpret as follows:  Let `out` be the key returned from the `L.Affine` call and `in` be the third argument input key.  (here `c`)  Then, abusing notation slightly `out.x = 1 * in.x + 0 * in.y + (-1)` and `out.y = 0 * in.x + 1 * in.y + 0`.  That is, `left_c` is just the cell displaced by `-1` in the x-direction.


```
hi_cells:NewFieldMacro('__apply_macro', L.Macro(function(c, x, y)
  return ebb `L.Affine(hi_cells, {{1,0, x},
                                  {0,1, y}}, c)
end))
lo_cells:NewFieldMacro('__apply_macro', L.Macro(function(c, x, y)
  return ebb `L.Affine(lo_cells, {{1,0, x},
                                  {0,1, y}}, c)
end))
local ebb shift_right_example_2( c : hi_cells )
  var left_c = c(-1,0)
  c.t_next = left_c.t
end
hi_cells:foreach(shift_right_example_2)
```
Usually, we won't type out the entire call to `L.Affine`.  Instead we'll use macros. (introduced in tutorial 15)  Ebb provides a special syntax overloading feature when a macro is installed with the name `'__apply_macro'`.  In this case, "function calls" to keys from the relation are redirected to the macro, which is supplied with the arguments as additional parameters.  This syntax is especially valuable for giving a shorthand for relative offsets in grid data.


```
lo_cells:NewFieldMacro('hi', L.Macro(function(c)
  return ebb `L.Affine(hi_cells, {{2,0, 0},
                                  {0,2, 0}}, c)
end))
local ebb down_sample( c : lo_cells )
  var sum_t = c.hi(0,0).t + c.hi(0,1).t
            + c.hi(1,0).t + c.hi(1,1).t
  c.t = sum_t / 4.0
end
local ebb up_sample( c : lo_cells )
  var d_t = c.t_next - c.t
  c.hi(0,0).t_next += d_t
  c.hi(0,1).t_next += d_t
  c.hi(1,0).t_next += d_t
  c.hi(1,1).t_next += d_t
end
```
The second macro we define here lets us access the higher-resolution grid from the lower resolution grid.  Using this connection, we can define routines to down-sample the current t-field, and also to up-sample and apply the diffusion results.  Notice how various access macros can be chained together.  We first access the high-resolution grid with `c.hi`, but then can immediately use the offset macro to locally navigate to the other 3 cells covered by the low resolution cell.


```
local timestep    = L.Constant(L.double, 0.25)
local conduction  = L.Constant(L.double, 0.5)
local friction    = L.Constant(L.double, 0.95)

local ebb diffuse_lo( c : lo_cells )
  var avg = (   c(1,0).t + c(-1,0).t
              + c(0,1).t + c(0,-1).t ) / 4.0
  var d_t = avg - c.t
  c.t_next = c.t + timestep * conduction * d_t
end
local ebb wave_hi( c : hi_cells )
  var avg = (   c(1,0).t + c(-1,0).t
              + c(0,1).t + c(0,-1).t ) / 4.0

  var spatial_d_t   = avg - c.t
  var temporal_d_t  = (c.t - c.t_prev)

  c.t_next = c.t + friction * temporal_d_t
                 + timestep * conduction * spatial_d_t
end
```
Now, we define the simulaiton at each resolution.


```
local sum_t     = L.Global(L.double, 0)
local max_diff  = L.Global(L.double, 0)
local ebb measure_sum( c : hi_cells )
  sum_t += c.t
end
local ebb measure_diff( c : hi_cells )
  var diff = L.fmax( L.fmax( L.fabs(c.t - c(0,0).t),
                             L.fabs(c.t - c(0,1).t) ),
                     L.fmax( L.fabs(c.t - c(1,0).t),
                             L.fabs(c.t - c(1,1).t) ))
  max_diff max= diff
end

local ebb visualize_hi( c : hi_cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var c = { L.xid(c), L.yid(c), 0 }
  vdb.point(c)
end
```
To define the visualization function, we use the debugging functions `L.xid()` and `L.yid()` which recover the numeric ids identifying which specific cell we're in.  We also define some debugging stats for the console.  In particular, we expect that since we defined our simulation carefully, we should preserve the sum of `t` and see a gradual decrease in the gradient as the diffusion behavior eventually dominates.


```
for i=1,200 do
  lo_cells:foreach(down_sample)

  lo_cells:foreach(diffuse_lo)
  hi_cells:foreach(wave_hi)

  lo_cells:foreach(up_sample)

  -- step forward
  hi_cells:Swap('t_prev','t')
  hi_cells:Swap('t','t_next')
  lo_cells:Swap('t_prev','t')
  lo_cells:Swap('t','t_next')

  vdb.vbegin()
  vdb.frame()
    hi_cells:foreach(visualize_hi)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    sum_t:set(0)
    hi_cells:foreach(measure_sum)
    hi_cells:foreach(measure_diff)
    print( 'iteration #'..tostring(i),
           'max gradient: ', max_diff:get()..'   ',
           'sum_t:',         sum_t:get() )
  end
end
```
Our simulation loop down-samples the field, runs both simulations, and then up-samples the diffusion results to combine with the wave simulation step.  Then we cycle our buffers, visualize and collect statistics.










## 17: Subsets

In tutorial 07, computing heat diffusion using the standard grid library, we imposed boundary conditions instead of having both directions be periodic.  To do so, we made use of subsets.  In this tutorial, we'll see how to define and use subsets of a relation.

```
import "ebb"

local vdb   = require('ebb.lib.vdb')

local N = 50

local cells = L.NewRelation {
  name      = 'cells',
  dims      = { N, N },
  periodic  = { false, false },
}

cells:NewFieldMacro('__apply_macro', L.Macro(function(c, x, y)
  return ebb `L.Affine(cells, {{1,0, x},
                               {0,1, y}}, c)
end))
```
We'll start the program by creating a cells relation without any periodicity, and define an offset macro.


```
local timestep    = L.Constant(L.double, 0.04)
local conduction  = L.Constant(L.double, 1.0)
local max_diff    = L.Global(L.double, 0.0)

cells:NewField('t', L.double):Load(function(xi,yi)
  if xi == 4 and yi == 10 then return 1000 else return 0 end
end)
cells:NewField('t_next', L.double):Load(0)

local ebb compute_step( c : cells )
  var avg_t = ( c(1,0).t + c(-1,0).t 
              + c(0,1).t + c(0,-1).t ) / 4.0
  var d_t   = avg_t - c.t
  c.t_next  = c.t + timestep * conduction * d_t
end
```
Here we define the basic simulation variables, fields, and core update function.

However, the update function has a problem.  If we simply execute `cells:foreach(compute_step)` then all of the cells on the boundary will try to access neighbors that don't exist, resulting in the equivalent of array-out-of-bounds errors.  These might manifest as segmentation faults, bad data, or in any number of other ways.


```
cells:NewSubset('interior', { {1,N-2}, {1,N-2} })
```
Instead of executing `compute_step()` over all the cells, we want to execute it only over the "interior" cells.  Ebb lets us define this _subset_ using the `NewSubset()` function.  We pass the function a name for the subset, and a list of (inclusive) ranges specifying a rectangular subset of the grid.


```
cells:NewSubset('boundary', {
  rectangles = { { {0,0},     {0,N-1}   },
                 { {N-1,N-1}, {0,N-1}   },
                 { {0,N-1},   {0,0}     },
                 { {0,N-1},   {N-1,N-1} } }
})
```
Instead of defining a subset by specifying a single rectangle, we can also specify a set of rectangles.  Here we use four rectangles to specify the left, right, bottom and top boundaries of the grid.  This is the complement of the 'interior' subset.


```
cells:NewFieldReadFunction('is_left_bd',   ebb (c) return L.xid(c) == 0 end)
cells:NewFieldReadFunction('is_right_bd',  ebb (c) return L.xid(c) == N-1 end)
cells:NewFieldReadFunction('is_bottom_bd', ebb (c) return L.yid(c) == 0 end)
cells:NewFieldReadFunction('is_top_bd',    ebb (c) return L.yid(c) == N-1 end)
```
Within the boundary, we want to be able to identify which side(s) a cell is on.  We hide these tests behind field functions so that the meaning of the code is more clear.


```
local ebb compute_neumann_boundary_update( c : cells )
  var sum_t = 0.0
  if not c.is_left_bd   then sum_t += c(-1,0).t
                        else sum_t += c.t end
  if not c.is_right_bd  then sum_t += c(1,0).t
                        else sum_t += c.t end
  if not c.is_bottom_bd then sum_t += c(0,-1).t
                        else sum_t += c.t end
  if not c.is_top_bd    then sum_t += c(0,1).t
                        else sum_t += c.t end
  var d_t = sum_t / 4.0 - c.t
  c.t_next  = c.t + timestep * conduction * d_t
end
```
A Neumann boundary condition specifies a zero-derivative at the boundary in the direction of the boundary.  That is, the flux is 0, or put another way, no heat should leave or enter the simulation. (We can test this.) We simulate this condition by having non-existant neighbors assume the same temperature value as the centered cell. (i.e. a difference of 0)

Notice that if we execute this function over all of the cells, we will compute the same result for interior cells as the `compute_step()` function.  Depending on a variety of factors in the implementation and hardware, this may be a more or less efficient approach.  (You can test the difference below)  If these branches contain much more math and we run on a GPU, then launching over seperate subsets is likely to be much more efficient.


```
local max_diff = L.Global(L.double, 0.0)
local sum_t    = L.Global(L.double, 0.0)

local ebb measure_diff ( c : cells )
  var diff = L.fmax( L.fmax( L.fabs(c.t - c(0,0).t),
                             L.fabs(c.t - c(0,1).t) ),
                     L.fmax( L.fabs(c.t - c(1,0).t),
                             L.fabs(c.t - c(1,1).t) ))
  max_diff max= diff
end
local ebb measure_sum ( c : cells )
  sum_t += c.t
end

local ebb visualize ( c : cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  vdb.point({ L.xid(c), L.yid(c), 0 })
end
```
visualization and statistics functions are defined above.


```
for i=1,20000 do
  cells.interior:foreach(compute_step)
  cells.boundary:foreach(compute_neumann_boundary_update)
  --cells:foreach(compute_neumann_boundary_update)
  cells:Swap('t','t_next')


  if i % 1000 == 0 then -- measure statistics and visualize every 1000 steps
    vdb.vbegin()
    vdb.frame()
      cells:foreach(visualize)
    vdb.vend()

    max_diff:set(0)
    sum_t:set(0)
    cells:foreach(measure_sum)
    cells:foreach(measure_diff)
    print( 'iteration #'..tostring(i),
           'max gradient: ', max_diff:get()..'   ',
           'sum_t:',         sum_t:get() )
  end
end
```
You can experiment with different parameters and methods for running this simulation loop here.















