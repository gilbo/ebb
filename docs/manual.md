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
local L = require 'ebblib'

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





# Getting Started Tutorials

See the [tutorials](tutorials.html) file for a sequence of example code.



# Overview

Ebb consists of two parts: an embedded language, and a Lua API.  The language proper is used to define Ebb _functions_, while the Lua API is used to construct and interrogate the data structures, as well as launch functions via `foreach` calls.  For instance, in the `hello42` sample program above, the `printsum()` function is written in the Ebb language, while the rest of the program makes calls to the API.

In addition to these two parts, a set of standard domain and support libraries are provided, which this documentation will also discuss.

The remainder of the manual will assume a passing familiarity with the structure of Ebb programs.  For a more intuitive introduction to the language, please see the [tutorials](tutorials.html).







# The Ebb Language

The Ebb language is used to define Ebb functions, which can either be used in other Ebb functions, or executed for each element of some relation.

-------------------------------------------------------------------------

## Ebb functions

Ebb functions can be declared anonymously in place of a Lua expression just like Lua and Terra functions
```
local foo = ebb() ... end
```
or syntactic sugar can be used to name and bind a function to a local or global/pre-existing name
```
ebb foo() ... end
local ebb bar() ... end
```

Arguments to an Ebb function can be supplied typed or un-typed
```
ebb foo ( x : L.double, y : L.int ) ... end
ebb bar ( z, another_arg) ... end
ebb baz ( x : L.double, y ) ... end
```

If no type is supplied for an argument, then the argument type will be inferred when the function is called.  A single function can be specialized multiple different ways, so if a second call produces a different type signature, then the function will be typechecked and compiled a second time.

Functions are processed at three distinct points.  When the Lua thread of control hits a definition, the function is defined, capturing any values from the enclosing Lua context.  Then, whenever a function is called for the first time, it is typechecked and compiled.  Finally, both the first and every subsequent time a function is called, it is executed.  As mentioned above, attempting to execute a function in a new context may cause a repetition of typechecking and compilation.

One important note here is that unless a function is used, it will not be type-checked.  This is unlike static languages, where all code is checked for errors at compile time, but also unlike dynamic languages in that the code is all type-checked on the first function execution, even those parts of the function that fail to execute.

Ebb functions consist of a sequence of statements, and possibly a final return statement
```
ebb foo( x : L.double, y : L.double )
  var z = 32.0
  return x + y * z
end
```

If a function takes exactly one argument, which is a key from some relation and returns no values, then that function can be executed for each element of the relation.  These functions are the primary (data-parallel) computations of Ebb
```
ebb dilate_vertex( v : vertices )
  var magnitude = L.length(v.pos)
  v.pos = v.pos * magnitude
end

vertices:foreach(dilate_vertex)
```

-------------------------------------------------------------------------

## Types, Literals, and Casting

Ebb types (detailed below) include
```
L.int    L.float    L.double    L.bool    L.uint64
```
as well as vectors and matrices of these primitive types and _key types_ representing references to rows of a given relation.  Whenever a relation `rel` is used in place of a type, it will be automatically promoted to the type `L.key(rel)`.

Literals use the same format conventions as C and Terra.  `0` is assumed to be an integer, `0.0` a double, and `0.0f` a float.  `0ULL` is a uint64.

In order to cast an expression in Ebb to a different type, that type can be invoked as if it were a function call.  For example, casting a double to an integer
```
ebb round( x : L.double )
  return L.int(x)
end
```

-------------------------------------------------------------------------

## Expressions

Ebb supports the following built in binary logic/arithmetic operations with their usual behavior:

```
  ==    ~=    <     >     <=    >=
  not   or    and   
  +     -     *     /     %
```

In addition the unary-prefix operations `-` and `not` are also supported.  Matching sized vectors or matrices can be added or subtracted, as well as being scaled or divided by a scalar.  However, matrix-vector multiplication and other matrix/vector operations are not built in.

Vectors can be constructed using curly braces, such as in this example function that constructs a vector with 3 copies of the argument
```
ebb vec_of_3 ( x )
  return { x, x, x }
end
```

Matrices can likewise be constructed by placing curly braces around 3 vectors of the same length
```
ebb mat_of_2 ( x )
  return { { x, x },
           { x, x } }
end
```
Subsequent vectors are assumed to be subsequent rows of the matrix.

Square brackets are used to access elements of vectors and matrices.
```
ebb matvec3 ( A, x )
  return { A[0,0] * x[0] + A[0,1] * x[1] + A[0,2] * x[2],
           A[1,0] * x[0] + A[1,1] * x[1] + A[1,2] * x[2],
           A[2,0] * x[0] + A[2,1] * x[1] + A[2,2] * x[2] }
end
```

Other Ebb functions are called like you would expect
```
ebb mat_of_3 ( x )
  return { vec_of_3(x), vec_of_3(x), vec_of_3(x) }
end
```

Likewise, Terra or C functions can be directly inlined into an Ebb function, though Ebb may or may not be able to port the resulting function between CPU and GPU without using a different version of the function.

```
terra sum_vals( x : double, y : double )
  return x + y
end

ebb double_val ( x : L.double )
  return sum_vals( x, x )
end
```

-------------------------------------------------------------------------

## Declarations and Assignments

Ebb variables can be declared with a type annotation and/or initialized.  If no explicit type annotation is provided, then the type is inferred from the intialization expression.
```
ebb foo()
  var x : L.double
  var y : L.int = 42
  var z = 3.2
end
```

Variables can have their values re-assigned
```
ebb foo()
  var x : L.double
  x = 3.5
  x = x * x
  return x    -- will return 7.0
end
```

-------------------------------------------------------------------------

## Field/Global Writes, Reads, and Reductions

Given a key from some relation, we can access the value of a field of the relation at a given element as if it were member data
```
ebb tet_mass ( t : tetrahedra )
  return t.volume * t.density
end
```
When accessed in this way, we say a field is being _READ_.

Alternatively, we can write functions that _WRITE_ to fields, as we demonstrate here by computing triangle normals
```
ebb compute_normal ( t : triangles )
  var n = L.cross( t.v[1].pos - t.v[0].pos,
                   t.v[2].pos - t.v[0].pos )
  t.normal = n / L.length(n)
end

Finally, functions can _REDUCE_ into fields using any of the reduction operators `+=`, `*=`, `min=`, or `max=`:
```
ebb newton_step ( v : vertices )
  v.pos      += 0.2 * v.velocity +
                0.2 * 0.2 * v.acceleration
  v.velocity += 0.2 * v.acceleration
end
```

Whenever an Ebb global variable is used on the right-hand-side of an an assignment, we say it is _READ_.
```
local uniform_density = L.Global(L.double, 0.2)
ebb tet_mass ( t : tetrahedra )
  return t.volume * uniform_density
end
```

Like fields, we can _REDUCE_ into Ebb globals, though Ebb functions are not allowed to write Ebb globals.
```
local max_vel = L.Global(L.double, 0.0)
ebb measure_max_vel ( v : vertices )
  max_vel max= L.length(v.vel)
end
```

-------------------------------------------------------------------------

## Phase Checking

Besides usual typechecking, Ebb implements a special kind of typechecking that we call _phase-checking_.  Phase-checking ensures that an Ebb function can be concurrently executed for each element of a relation without the possibility of data-races.  It consists of three simple rules

* If a field is _exclusively_ accessed through the _centered_ key (defined below) then any combination of _READs_, _WRITEs_, and _REDUCEs_ are allowed.

* If a field is accessed dependently (through a key-field, query-loop, or affine-indexing) then either the field must only be _READ_, or only _REDUCED_ with a consistent reduction operation.

* If a global is accessed, it must only be _READ_, or only be _REDUCED_ with a consistent reduction operation.

The _centered_ key is the key passed in from the relation on which `rel:foreach(func)` is called.  For instance, in the following execution `v` is centered, while `nv` is not, even though `nv` is also from `vertices`.  Because `vertices.t` is not _exclusively_ accessed through the centered key, we can only _READ_ from it, not reduce or write, even via the centered key.
```
local ebb compute_update( v : vertices )
  var avg = 0.0
  for e in v.edges do
    var nv = e.head
    avg += e.weight * (nv.t - v.t)
  end
  -- the following line would cause a phase-checking error
  -- v.t += avg
  v.t_change = avg
end

vertices:foreach(compute_update)
```

-------------------------------------------------------------------------

## Control Flow

Ebb supports `do ... end` blocks in order to hide variables in local scopes.
```
ebb foo( x : L.double )
  x = 2.0
  var y = 3.0
  do
    x = 4.0
    y = x
  end
  L.print(x, y) -- will output 2.0, 4.0
end
```

Ebb supports standard `if` statements as well.
```
ebb sgn( x : L.double )
  var val : L.double
  if x > 0.0 then
    val = 1.0
  elseif x < 0.0 then
    val = -1.0
  else
    val = 0.0
  end
  return val
end
```

Ebb supports `for` loops with numeric bounds.  Unlike Lua, but like Terra, Ebb for loops count over the range exclusive of the upper bound.  That is, the loop below will count `2, 3, ..., n-1, n` but not `n+1`.
```
ebb fact( n : L.int )
  var prod = 1
  for k=2,n+1 do
    prod *= k
  end
  return prod
end
```

Ebb supports `while` loops as well
```
ebb fact( n : L.int )
  var prod = 1
  while n > 1 do
    prod *= n
    n = n - 1
  end
  return prod
end
```

And Ebb supports the Lua/Terra `repeat ... until <cond>` loop
```
ebb double_to_above( n : L.int, lower : L.int )
  repeat
    n *= 2
  until n >= lower
  return n
end
```

Finally, Ebb supports the query loop, which usually will look something like the following
```
ebb compute_local_avg( v : vertices )
  var sum = 0.0
  var n   = 0
  for e in v.edges do
    sum += e.head.t
    n   += 1
  end
  v.avg = sum / n
end
```

Though, as explained in the tutorials, `v.edges` is the query expression `L.Where(edges.tail, v)` hidden behind a macro.



-------------------------------------------------------------------------

-------------------------------------------------------------------------









# The Ebb API

In the following, we assume that the Ebb library has been loaded into a local variable `L`, e.g. via the command `local L = require 'ebblib'`.


## Types

Ebb types are Lua objects, available via the standard library.  We can test whether a Lua value is an Ebb type by calling

```
L.is_type(val)
```

There are 5 primitive types

```
L.bool    L.int    L.float    L.double    L.uint64
```

We can test whether a type is primitive or not using

```
typ:isprimitive()
```

-------------------------------

Small matrices or vectors of primitives are given types via the constructors

```
L.vector(primitive_type, n_entries)
L.matrix(primitive_type, n_rows, n_columns)
```

We can test whether a type is a vector, matrix, or neither via the calls

```
typ:isvector()
typ:ismatrix()
typ:isscalar() -- neither of the above
```

If a type is a vector, then we can get the number of entries as `typ.N`; if a type is a matrix, then we can get the number of rows as `typ.Nrow` and number of columns as `typ.Ncol`.  We can retrieve the primitive type by calling

```
typ:basetype()
```

which can also be called on primitives, returning the original type.

As a convenience, several standard vector and matrix types are given shorthands by the standard library.

```
L.vec2d     -- is L.vector(L.double, 2)
L.mat3f     -- is L.matrix(L.float, 3, 3)
L.mat2x3i   -- is L.matrix(L.int, 2, 3)
```

These shorthands will work for 2, 3, and 4-dimensional types, and use the primitive type coding that d=double, f=float, i=int, and b=bool.

We can also apply the following tests irrespective of whether a type is scalar, vector or matrix

```
typ:isintegral() -- true for L.int and L.uint64
typ:isnumeric() -- true for all but L.bool
typ:islogical() -- true for only L.bool
```

-------------------------------

Lastly, we can construct _key types_ to talk about references between relations.

```
L.key(relation)
```

To inquire for the specific relation, we can access `typ.relation`.

As a convenience, the Ebb API will automatically promote a relation `rel` into the type `L.key(rel)` anywhere it expects to get a type.  For instance, we can construct vectors and matrices of keys, and pass a relation directly in place of the primitive.

```
L.vector(L.key(vertices), 3)
-- equivalently
L.vector(vertices, 3)
```

We can test whether a type is a scalar key using the function

```
typ:isscalarkey()
```

Alternatively, we can test whether it is any sort of key type: scalar, vector or matrix

```
typ:iskey()
```

### Summary of Types

Scalar types are either one of 5 primitives or a key-type.  All of these scalar types can be organized into vectors and matrices.

If type-casting of values in code can be done without any loss of precision, Ebb will insert coercions as necessary.


--------------------------------------------------------------------------

## Builtins

Ebb defines built-ins for two reasons: 1) some special functionality cannot be defined as user-level libraries; 2) to allow portability of the standard math routines.  (Note: We hope to move the math functions into user-level libraries in the future)

Given some arbitrary Lua value, we can test whether it's an Ebb builtin using the function

```
L.is_builtin(val)
```

Ebb provides assert and print functions that are portable across CPU and GPU execution to help with debugging.  The `L.print()` built-in supports a small subset of the standard printf functionality.  If you find the print function inadequate for your debugging needs, please let the Ebb developers know.

Note that both `L.assert()` and `L.print()` may cause side-effects.  When executed in parallel, no guarantees are made about the order in which these side-effects take place.  Therefore, these functions should only be used for debugging.  Field dumping should be used when reproducible output is desired.

```
L.assert(booltest)
L.print(...)
```

While the internal representation of key values is hidden from Ebb programmers, we allow users to extract stable identifiers for rows, which can also be helpful for debugging:

```
L.id(keyval)  -- return an identifier for an element of a non-grid relation
L.xid(keyval)
L.yid(keyval)
L.zid(keyval) -- return coordinates for an element of a grid relation
```

The "topological query" functions (`L.Affine()` and `L.Where()`) are also builtins, although they are discussed [elsewhere in this manual](BROKEN)

TODO: WHERE?

We provide the 3 following convenience functions for numeric vectors

```
L.length(vec)
L.cross(vec_a, vec_b)
L.dot(vec_a, vec_b)
```

and a GPU portable random number generator that returns a double in the range 0 to 1.

```
L.rand()
```

We also provide a set of functions from _math.h_.  When executed on a GPU, standard CUDA implmentations or special GPU instructions will be used in their place.  If you need access to standard math functions you don't see here, please contact the Ebb developers.

```
L.acos(x)
L.asin(x)
L.atan(x)
L.cbrt(x)
L.ceil(x)
L.cos(x)
L.fabs(x)
L.floor(x)
L.fmax(x,y)
L.fmin(x,y)
L.fmod(x,denom)
L.log(x)
L.pow(base,exp)
L.sin(x)
L.sqrt(x)
L.tan(x)
```

### External C functions in Ebb code

Lastly, any C-functions imported via Terra can be used in Ebb code as if they were a built-in.  (For instance, all of the C `math.h` library can be imported this way.)  However, functions imported this way will usually not be portable to GPU.


--------------------------------------------------------------------------

## Globals

Global variables are used to represent non-spatial values (i.e. not defined per-element) that change over the course of a simulation.  If you're instead confident that the value will remain fixed, using a `Constant` instead will  result in better performance.

Global variables are created using the following function call

```
local glob = L.Global( typ, init_value )
```

A lua value can be tested for whether or not it's an Ebb global using

```
L.is_global(val)
```

Besides using a global inside Ebb functions, we can set or get its value from Lua using

```
glob:set(new_value)
local lookup_value = glob:get()
```

And finally, we can inquire for a Global's type using

```
local typ = glob:Type()
```

Inside an Ebb function, a global can either be read from, or reduced into.

--------------------------------------------------------------------------

## Constants and Literals

Constants work very similarly to globals, except the value of a constant cannot be modified from Lua or from Ebb.

```
local const = L.Constant( typ, init_value )

L.is_constant(val)

local lookup_value = const:get()

local typ = const:Type()
```

When used in Ebb code, a constant will be assigned the specified type.

Literals in Ebb code can arise from a literal in the code, such as the `1.0` in the following example.  In this case, Ebb interprets `1.0` as a literal of type `double` and so coerces `x` into a double before taking the sum.
```
local ebb literal_example_1( x : int )
  return x + 1.0
end
```

However, literals can also arise from inlined Lua variables, such as `inc` in the following example.  However, because the untyped value `1` is inlined into the code (all Lua numbers are of the single Lua type _number_) Ebb chooses the most conservative possible type for the literal, which is `int` here.  Consequently, the function returns an `int` rather than a `double`.
```
local inc = 1.0
local ebb literal_example_2( x : int )
  return x + inc
end
```

Constants can solve this problem with literals by allowing the programmer to provide an explicit type for a value declared in Lua.


--------------------------------------------------------------------------

## Relations

Apart from globals, (and data stored only at the Lua level) all Ebb data is stored in _relations_, which are like database/spreadsheet tables.  Usually a domain library will set up relations for programmers, though domain libraries do not have any kind of privilleged API access.

The following call creates a new relation with `n_size` rows/elements.  We specify the `name` as well for debugging output.
```
local relname = L.NewRelation{
  name = 'relname',
  size = n_size,
}
```

If we want to instead declare a 2d or 3d grid-structured relation, then we instead call the following.  `dims_list` is a Lua list of 2 or 3 numbers; `periodic_list` is a list of 2 or 3 booleans.  If the `periodic` argument is ommitted, (it's optional) then by default no dimensions are periodic.
```
local relname = L.NewRelation{
  name      = 'relname',
  dims      = dims_list,
  periodic  = periodic_list,
}
```

We can test whether a lua value is a relation using the function
```
L.is_relation(val)
```

and given a relation, we can test whether it's a grid using
```
rel:isGrid()
```

We can retrieve any of the arguments we used to create the relation using the following functions
```
rel:Name()
rel:Size()
rel:Dims()
rel:Periodic()
```
If a relation is grid-structured, then `rel:Size()` returns the total number of elements.  If a relation is not grid-structured, then `rel:Dims()` returns a list with one number in it: the size of the relation.

The following call will print information about the relation and all the fields defined on it.  This function is not optimized for efficiency.  It should only be used for debugging and on relatively small relations.
```
rel:Print()
```

----------------------------

The following call will execute an Ebb function `efunc` for each element of a relation `rel`.
```
rel:foreach(efunc)
```

Besides defining fields (detailed below), macros or functions can be installed in place of fields.  A field-macro can be installed via the following call.
```
rel:NewFieldMacro(field_name, macro)
```
Field macros will only be passed a single argument: the key on which they get invoked.

There is one special exception to this rule.  If a field-macro is installed with the name `__apply_macro`, then it will get invoked whenever the syntax `k(arg1, arg2, ...)` is used.  The macro will get passed `k` as its first argument, followed by all of the arguments within the parentheses.

Besides field-macros, field-functions can be installed using one of the following three calls.  All calls take a field name, and an ebb function.  The reduce call also takes a (string) argument to specify which reduction operation is being overloaded.  Ebb functions installed as read-functions should take a single argument: the key on which they are invoked.  Ebb functions installed as write-functions or reduce-functions should take two arguments: the key on which they're invoked and the right-hand-side value that should be written or reduced.
```
rel:NewFieldReadFunction(field_name, efunc)
rel:NewFieldReduceFunction(field_name, reduce_op, efunc)
rel:NewFieldWriteFunction(field_name, efunc)
```

----------------------------

### Grouping and Query-Loops

If a relation `rel` is not grid structured, and has a field `kf` of scalar key type `L.key(r_target)`, then we can group `rel` by `kf` using the call
```
rel:GroupBy(rel.kf)
```
or equivalently
```
rel:GroupBy('kf')
```
Strings will be automatically converted into full fields.

Once `rel` is grouped by `kf`, and given a key `rt` from `r_target`, then we can execute the query-loop
```
for r in L.Where(rel.kf, r_t) do ... end
```
inside of ebb functions.

We can test whether a relation is grouped using the call
```
rel:isGrouped()
```
And given that a relation is grouped, we can query for the field it's grouped on using
```
rel:GroupedKeyField()
```
which will return the field object (not name string) of the key field `rel` is grouped by.

If a relation is neither grid-structured nor grouped, then we say that it's in _plain_ mode.  We can test for whether a relation is plain using
```
rel:isPlain()
```

--------------------------------------------------------------------------

## Fields

New fields can be defined on relations using the following call
```
rel:NewField(name, typ)
```
which returns the new field object.

Given an arbitrary Lua value, we can test whether it's a field using the call
```
L.is_field(val)
```

We can retrieve the field's name, parent relation, full name (parent relation's name concatenated with field name), type and size (i.e. the size of the parent relation) all using a set of functions

```
field:Name()
field:FullName()
field:Relation()
field:Size()
field:Type()
```

The contents of the field can be printed, though the `Print()` function is not optimized for efficiency.  It should only be used for debugging and on relatively small relations.
```
field:Print()
```

Given two fields with names `f1` and `f2` on the same relation `rel` and with the same type, we can swap their contents using the function
```
rel:Swap('f1', 'f2')
```

If we would like to copy the contents of `f1` to `f2` instead, we can call
```
rel:Copy{ from = 'f1', to = 'f2' }
```

-------------------------

In order to get data into and out of fields, we use the functions
```
field:Load(...)
field:Dump(...)
```
which are polymorphic based on their arguments.

When passed a value of the same type as the field, `Load()` will load that same value in for the field's value at every element.  For instance, the following load a double, and vector of 3 doubles respectively
```
f1:Load(3.2)
f2:Load({0,0,0})
```

If another field of the same type is passed to `Load()`, then the values are copied over from that field to initialize the first one.
```
f1:Load(f2)
```

If a Lua function is passed in as the argument, then the lua function will be invoked once per-argument being passed the id of the element and returning the value to be assigned for that element.  For instance, the following call will assign values equal to the log of the element's id to each element.
```
f1:Load(function(id) return math.log(id) end)
```
If the relation is a grid, then 2 or 3 identifiers are supplied instead.

If a Lua list of `field:Size()` values is passed to `Load()`, then those values will be used to initialize the field.  In the case of a grid, this should be a list-of-lists, reflecting the grid structure.
```
field:Load( { val1, val2, ... } )
grid_field:Load( { { val1, val2, ... }, ... })
```

Finally, loading can be perfomed using C or Terra functions, as detailed below in the section on File I/O

If `Dump()` is called with an empty table passed in as the argument, then `Dump()` will return a Lua list containing the values of the field.
```
local vals = field:Dump({})
```

Otherwise, we recommend using the File I/O facilities to dump the contents of a field to file.


--------------------------------------------------------------------------

## Subsets

To define a subset on a relation, we call the new subset function.

Currently, the API supports two ways of defining subsets of grids.  If you need other ways to specify subsets, please contact the developers.

The first form of `NewSubset()` takes a list of ranges specifying a rectangle, such as
```
rel:NewSubset(name, { {x_lo, x_hi}, {y_lo, y_hi}, {z_lo,z_hi} })
```
where `x_lo` and `x_hi` specify an inclusive range.  In the case of a 2d grid, the z-bounds should be omitted.

The second form of `NewSubset()` works similarly, but takes a whole list of rectangles in the form given for the first call
```
rel:NewSubset(name, {
  rectangles = { rect_1, rect_2, ... }
})
```
The subset is then defined as the union of these rectangles.

We can test whether a relation has any subsets defined using
```
rel:hasSubsets()
```

We can test whether a given Lua value is a subset object using
```
L.is_subset(val)
```

Given a subset, we can interrogate it for its name and parent _relation_
```
subset:Name()
subset:FullName()
subset:Relation()
```
`FullName()` will return a string that concatenates the parent relation's name and the subset's name with a '.' in-between.

Most importantly, ebb functions can be executed for each element in a subset using the following call
```
subset:foreach(efunc)
```


--------------------------------------------------------------------------

## Data Layout Descriptors

In order to support interoperability with externally defined code, Ebb defines a kind of metadata called a _data layout descriptor_ (DLD) to specify exactly where and how a field of data is laid out.  To get immediate access to the backing data regardless of where it is, call

```
local dld = field:GetDLD()
```

This will return a Lua table reflecting the C-struct format.  When passing this descriptor into a C or Terra function called from Lua, you should convert the descriptor into a C-struct by calling

```
dld:toTerra()
```

To get access to the DLD type via Terra, you can require the standard library
```
local DLD = require 'ebb.lib.dld'
```
which will then expose the Terra type as
```
DLD.C_DLD
```
Alternatively, if you are separately compiling C-code to link against an Ebb program, you can use the provided header file at [include/ebb/lib/dld.h](../include/ebb/lib/dld.h).  Both the `.t` and `.h` file include full documentation inline of the format.

A DLD is laid out as
```
typedef struct DLD {
  uint8_t         version[2];         /* This is version 1,0 */
  uint16_t        base_type;          /* enumeration / flags */
  uint8_t         location;           /* enumeration */
  uint8_t         type_stride;        /* in bytes */
  uint8_t         type_dims[2];       /* 1,1 scalar; n,1 vector; ... */

  uint64_t        address;            /* void* */
  uint64_t        dim_size[3];        /* size in each dimension */
  uint64_t        dim_stride[3];      /* 1 = 1 element, not 1 byte */
} DLD;
```

The `address` field holds a pointer to the start of the data.

To illustrate how the different parameters of the description work, suppose we have a 3d grid of 3-by-3 small-matrices of doubles.  To access the `i,j`-th entry in the small matrix at grid cell `x,y,z`, we would compute the following address.

```
uint elem_index     = x * dld.dim_stride[1] +
                      y * dld.dim_stride[2] +
                      z * dld.dim_stride[3] ;
double * elem_addr  = (double*)( (uint*)(dld.address) +
                                 elem_index * dld.type_stride );
double * entry_addr = elem_addr + i*3 + j;
```

Note the following. The `dim_stride` values specify strides in elements not bytes.  Elements are assumed to be stored `type_stride` bytes apart.  And small matrices are always stored in row-major order, densely packed.  Likewise vectors are densely packed, though the `type_stride` may specify that there should be some number of unused padding bytes at the end of a vector or matrix element to improve alignment, e.g. for SIMD-vectorization.

The `location` field either assumes the value `CPU` or `GPU` (defined as constants in both the `.t` and `.h` files).

The `base_type` enumeration is a bit more complicated, but constants are defined for all of the following:

```
UINT_8   UINT_16  UINT_32  UINT_64  
SINT_8   SINT_16  SINT_32  SINT_64  
FLOAT    DOUBLE
```

Additionally, since Ebb maintains specialized key encodings, `dld.t` defines constants for all possible key encodings, e.g.
```
KEY_32   KEY_64_32   KEY_16_32_64
```
In `dld.h` there are instructions for how to compose similar constants in a C file.

--------------------------------------------------------------------------

## Load and Dump (File I/O)

Ebb uses DLDs to support efficient, custom file I/O libraries.  This behavior is triggered by passing a Terra or C function `tfunc` to `Load()` or `Dump()`
```
field:Load( tfunc, ... )
field:Dump( tfunc, ... )
```
The first argument of `tfunc` should always have type `&DLD.C_DLD`.  Any remaining arguments passed into `Load()`/`Dump()` after `tfunc` will be passed into `tfunc` as additional arguments.  Similarly, any value returned from `tfunc` will be returned by the `Load()`/`Dump()` call.

For instance in the following example, we can pass in a filename and return an error code if the file doesn't exist.
```
local terra demo_load ( dld : &DLD.C_DLD, filename : &int8 )
  if not file_exists(filename) then
    return 1
  else
    return 0
  end
end

local errcode = field:Load(demo_load, "some_file.data")
```

Sometimes file formats require simultaneous access to multiple fields of a relation.  Ebb supports multi-field `Load()` and `Dump()` calls for these cases.  To load or dump fields `f1`, `f2`, etc. call
```
rel:Load( { f1, f2, ... }, tfunc, ... )
rel:Dump( { f1, f2, ... }, tfunc, ... )
```
`f1`, `f2`, etc. can be either field objects from `rel` or strings specifying fields of `rel`.  Just like the `field:Load()`/`field:Dump()` forms, any additional arguments are passed through to the Terra/C function, and any return value is returned.  When using this form the first DLD pointer argument will point to an array of DLDs: one for each field requested.

---------------------

Using Terra's metaprogramming facilities, it's possible to generic, optimized file I/O routines.  In order to support the necessary introspection to make this happen, Ebb allows for the creation of `Loader` and `Dumper` objects.
```
local loader = L.NewLoader(lua_func)
local dumper = L.NewDumper(lua_func)
```

These loader/dumper objects can be passed to field load/dump calls
```
field:Load(loader, ...)
field:Dump(dumper, ...)
```
When called like this, the Lua function used to create the loader or dumper is called with `field` as the first argument, and the remaining arguments passed through.  Any returned values, will be returned similarly.

Loader/Dumpers are used to implement the standard CSV library, which then allows simulation code to simply write
```
local CSV = require 'ebb.io.csv'

field:Load(CSV.Load)
field:Dump(CSV.Dump)
```
without having to worry about what type of field is being processed.


Like with other Ebb objects, we can test whether arbitrary Lua values are loader or dumper objects using the functions
```
L.is_loader(val)
L.is_dumper(val)
```

--------------------------------------------------------------------------

## Macros and Quotes

Ebb macros work similarly to Terra macros, though they are less developed.  Macros consist of Lua functions that are executed during typechecking/compilation.  These macro functions are passed in Ebb code as arguments, and expected to return a piece of quoted Ebb code.
```
local macro = L.Macro(lua_func)
```

We can test whether a given lua value is an Ebb macro using
```
L.is_macro(val)
```

Ebb quotes can be constructed in two forms.
```
local q = ebb \`some_ebb_expression_here
local q = ebb quote
  some_ebb_statement_1
  some_ebb_statement_2
  ...
in
  some_ebb_expression_here
end
```
The second form is converted to a let-expression internally, while the first form is substituted in directly.  For instance, the first form allows writing `ebb `L.Where(...)` even though `L.Where(...)` is neither an expression nor statement in the Ebb language. (It can only ever occur in a query-loop)

WARNING:  Like macros in most languages, if you use the argument to a macro more than once, you will create duplications of the subtrees.  If you do not intend to do this, you need to assign the value of a macro argument to a temporary using the `ebb quote ... in ... end` form.  In general, please rely on Ebb functions in preference to macros where possible.

If you find the current macro system insufficient or buggy for something you want to accomplish, please contact the developers about possible changes.


--------------------------------------------------------------------------

--------------------------------------------------------------------------

--------------------------------------------------------------------------








