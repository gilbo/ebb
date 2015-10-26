% Liszt-Ebb Manual

# Liszt-Ebb Manual

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




# Getting Started -- Tutorials (Writing Code by Example)

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
Another possibile variation is that we could write the computation as a per-edge, rather than a per-vertex computation, though we need to pre-compute and store each vertex's degree and zero the destination before reducing.


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





## 07: Using Grids










## 08: Coupling Particles to a Grid









## 09: Custom Domain Modeling






## Other Examples












# Full Manual


## Functions

Tenslang allows programmers to write simple straight-line code functions.  We can define one of these functions either anonymously, inline, or in a statement

```
-- define a function as a global symbol in Lua
tensl foo()
  return 42
end

-- define a function as a local symbol
local tensl bar()
  return 42 + 1
end

-- define a function anonymously and assign to a variable
local baz = tensl()
  return 42 + 2
end
```

Functions can be given arguments, but those arguments must be typed

```
local tensl scale_and_add_one( s : TL.num, x : TL.num )
  return s*x + 1
end
```

(notice that we need to look up Tenslang types in the Tenslang standard library)

Tenslang functions can also return multiple values.

```
local tensl some_pi()
  return 3, 1, 4, 1
end

local a, b, c, d = some_pi()
print(a,b,c,d) -- will print    3   1   4   1
```

While Tenslang lacks fully-recursive functions, you can call functions from
other functions.

```
local tensl square( x : TL.num )
  return x * x
end
local tensl quad( x : TL.num )
  return square(square(x))
end
assert(quad(2) == 16)
```

## Types, Literals, Constants, Values

### Types
Tenslang has two primitive types
```
TL.num TL.bool
```

In addition to primitive types, Tenslang values can also be tensor-typed.
(For the unfamiliar, tensors are simply a generalization of vectors and
matrices)  Many common tensor types are given explicitly aliases.
For instance,
```
TL.vec3     -- a vector of 3 numbers
TL.mat2b    -- a 2x2 matrix of bools
TL.mat2x3   -- a 2x3 matrix of numbers
```

We can also construct vector and matrix types using the more general forms.
```
TL.vector(primitive_type, n)
TL.matrix(primitive_type, n, m)
```
These are just special cases of the more general tensor constructor.
```
TL.tensor(primitive_type, ...)
```

Notice that all tensor types must have a constant fixed dimension.  Tensors are intended to encode small, constant sized data that show up regularly
when dealing with geometry, simulation and graphical data. (e.g. points, colors, linear/affine transformations, stress matrices, etc.)  Trying to use these types to store large vectors of data will result in very bad performance in almost all cases.

### Literals and Constants
Numeric literals can be written into Tenslang code as you would expect.

You can also write tensor literals into code
```
{1,2,3} -- a TL.vec3
{{1,0,0},{0,1,0},{0,0,1}} -- a TL.mat3
```

Aside from literals, constant values can also be computed in Lua and then
captured by a Tenslang function definition.  However, to allow programmers to be more explicit, we also provide Lua-level constants.

```
local answer  = TL.Constant(TL.num, 42)
local idmat3  = TL.Constant(TL.num, {{1,0,0},{0,1,0},{0,0,1}})

local tensl foo()
  return idmat3[0,0], answer
end
```

### Casting

Tenslang supports explicit typecasting.  This can be used to cast between boolean and numeric values or to explicitly cast tensor indices into numbers (see below).

To perform an explicit typecast, you just use a type as if it were a function.
```
local tensl tobool( x : TL.num )
  return TL.bool(x)
end

local tensl toboolmat( x : TL.mat3b )
  return TL.mat3b( x )
end
```


## Variables and Assignment

Aside from function arguments, Tenslang allows for the declaration of
variables.  A variable must either be explicitly typed or assigned
an initial value (from which the type can be inferred).

```
local tensl foo()
  var x : TL.num
  var y = 0.0
  var z : TL.num = 1.0
  return y + z
end
```

Assignments to variables are also allowed, including to multiple variables at once.  This can be used to capture multiple return values from a function call.

```
local tensl square_triple(x : TL.num)
  var square : TL.num
  var triple : TL.num
  var a = x * x
  square, triple = a, a*x
  return square, triple
end

local tensl foo()
  var x : TL.num
  var y : TL.num
  x,y = square_triple(3)
end
```

However, the current version of Tenslang does not support the declaration and assignment of multiple variables at the same time.

## Control Flow

The scope of variables can be limited with do blocks, though other kinds of control flow are omitted from the language.

```
local tensl shadow()
  var x = 3
  do
    var x = 5
  end
  return x
end

assert(shadow() == 3)
```


## Primitive Expressions

Tenslang supports the following kinds of operations between primitive expressions.

```
local tensl arithmetic_plus()
  var x  = 12
  var y  = 32
  var bt = true
  var bf = false

  var negx    = -x
  var notb    = not bt

  var bor     = bt or bf
  var band    = bt and bf

  var lt      = x < y
  var gt      = x > y
  var lte     = x <= y
  var gte     = x >= y
  var eq      = x == y
  var neq     = x ~= y    -- notice Lua style not-equal syntax

  var sum     = x + y
  var diff    = x - y
  var prod    = x * y
  var quot    = x / y

  var parens  = x * (x + y)
end
```

### Built-in functions

There is a built-in assertion function `TL.assert()` that's useful for writing testing code.  If the argument evaluates to false, then the entire program will terminate.

```
local tensl runtest()
  var answer = 42
  TL.assert(answer / 2 == 21)
end
```

### External Functions

You can extend Tenslang with custom external functions, provided they follow
a Terra calling convention (e.g. C functions).  To do this, the `TL.extern()` library function is provided.

For example, suppose you want to use a sqrt function.  You could import the C math library using Terra and then bind the function in using `TL.extern()`
```
local cmath = terralib.includecstring [[#include "math.h"]]
local sqrt = TL.extern('sqrt', TL.arrow(TL.num, TL.num), cmath.sqrt)
```
The first argument to `TL.extern` is the function name for debugging purposes.
The second argument is the function's type, and the third argument is the implementation of the function.

For reference, you should assume that the `TL.num` type is represented with the C/Terra type `double`.

WARNING: In general, Tenslang assumes that functions imported in this way are "referentially transparent" in the sense that they should be indistinguishable from a deterministic state-free function _from the perspective of Tenslang_.  This allows that the implementation of an external function could still use side-effects or randomness to produce the same result more efficiently, or to record profiling information, etc.


## Tensor Expressions

### Tensor-Indexing Expressions

Tenslang's most sophisticated feature is its tensor system.  Most notably,
this includes its special _tensor-indexing_ expressions.  To understand these more intuitively, consider the following expression for the matrix-vector dot product.

```
local tensl matvecprod( m : TL.mat3, x : TL.vec3 )
  return :[i] +[j] m[i,j] * x[j]
end
```

In more words, this expression says "map over the index `i`, and within that sum over the index `j`, and within that take the product of the `i,j`-th entry of `m` with the `j`-th entry of `x`".

To make this even more intuitive, consider this expression for the dot product:
`+[i] x[i] * y[i]`
Imagine we replaced the `+[i]` with a big summation like a mathematician would write.  Then, we would have exactly the mathematical definition of the dot product.

To make the `:[i]` example clearer, consider the case of scaling a vector by `3`: `:[i] 3 * x[i]`.

As is apparent from the matrix-vector multiplication example, these two _tensor-indexing_ expressions can be combined to express more complicated forms.  In the following code-snippet, you can see how a wide variety of matrix-vector operations can all be expressed from these two simple constructions.

```
var dot_product   = +[i] x[i] * y[i]

var mat_vec_prod  = :[i] +[j] M[i,j] * x[j]

var transpose     = :[i,j] M[j,i]

var inner_product = +[i,j] x[i] * Q[i,j] * y[j]

var outer_product = :[i,j] x[i] * y[j]

var mat_mat_prod  = :[i,j] +[k] A[i,k] * B[k,j]

var trace         = +[i] M[i,i]

var frobenius     = +[i,j] M[i,j] * M[i,j]

var first_column  = :[i] M[i,0]

var sum_columns   = :[i] +[j] M[i,j]
```

Part of the magic of tensor-index expressions is that the appropriate range of the index variables (`i` in `:[i]` or `+[i]`) can be inferred by the type-checker.  In order for this to work, these _tensor-indicex_ variables are given a special type.  If you want to get their numeric value for computation with, you may need to explicitly cast the variable, at which point it can no longer be used to infer its dimension correctly.  Generally this is a minor detail.  The typechecker will complain if something is wrong.

In addition to the summation reduction, Tenslang also supports a multiplication reduction.


### Tensor-Construction Expressions

In addition to the above, tensors can also be constructed out of individual values in code, which always gives a way of more explicitly writing down a detailed expression.

```
var prefix_sum    = { x[0], x[0]+x[1], x[0]+x[1]+x[2] }

var cross_matix   = {{     0, -x[2],  x[1] },
                     {  x[2],     0, -x[0] },
                     { -x[1],  x[0],     0 }}

var repeat_column = {x,x,x}
```




# Tenslang Lua API

Since Tenslang is embedded in Lua, the functions, constants and types are all represented as Lua objects/tables.  In order to help Lua scripts work with these objects, we provide a set of functions that one can introspect with.


## Constant Introspection

To test whether a Lua value is a Tenslang constant
```
TL.isconstant( obj )
```

To get the value of the constant
```
local val = constant:get()
```

To get the type of the constant
```
local typ = constant:gettype()
```


## Function Introspection

To test whether a Lua value is a Tenslang function
```
TL.isfunction( obj )
```

To test whether a Tenslang function has been compiled yet
```
func:iscompiled()
```

To force compilation without calling the function
```
func:compile()
```

To get a string containing the function's declared name
```
local name = func:getname()
```

To get the type of the function
```
local typ = func:gettype()
```

Note that functions have a special `TL.arrow(...)` type that cannot be used as a value inside of Tenslang programs.  However, arrow types can be constructed by passing a Lua list of argument types and Lua list of return types to the arrow constructor.  e.g. `TL.arrow({TL.num},{TL.mat3, TL.vec3})` is a function type for a function that takes one `num` argument and returns a matrix and vector as return values.  A Tenslang function that neither takes any arguments, nor returns any values has type `TL.arrow({},{})`.



## Type Introspection

To test whether a Lua value is a Tenslang type
```
TL.istype( obj )
```

If an object is a Tenslang type, then you can test for what kind of type it is with the following functions
```
typ:isvalue() -- any primitive or tensor type
  typ:isprimitive()
  typ:istensor()
    typ:isvector()
    typ:ismatrix()

typ:isarrow()
```

Tenslang ensures that all types are resolved to the same object, which means it's safe to compare two types using an equality check
```
assert(TL.vec3 == TL.vector(TL.num, 3))
```

To get the underlying primitive type for a tensor-type
```
local primtyp = tensortyp:basetype()
```
If you have a tensor type, then you can also get a Lua list of its dimensions `tensortyp.dims`, from which you can compute how many values there are in what shape.  Tenslang lays these out in row-major order in memory and expects similar nesting throughout the language.

Any value type, primitive or tensor can also be tested for what sub-class of value it is using the following tests.
```
valtyp:isnumeric()
valtyp:islogical()
```

The argument and return lists for arrow types can be extracted using the following two functions
```
local argtyps = arrowtyp:argtypes()
local rettyps = arrowtyp:rettypes()
```

## Static Function Compilation

To compile out a set of Tenslang functions into an object file
```
TL.compiletofile(object_filename, header_filename, {
  fname1 = function1,
  fname2 = function2,
  ...
})
```
`object_filename` and `header_filename` are strings with filesystem paths specifying where the results should be placed.  If `header_filename` is `nil` then no header will be generated.  The third argument is a table of functions to be exposed as visible symbols in the object file.  The table keys allow you to use an alternate function name when exporting/compiling in this way.

See the [`example_c/`](../example_c) directory for an example of `TL.compiletofile()` use.


