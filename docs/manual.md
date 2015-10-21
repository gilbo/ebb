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

For the rest of this manual, we assume a passing familiarity with the [Lua language](http://www.lua.org/).  Specifically, Ebb is embedded in Lua 5.1, though the Lua language is largely feature-stable today.  You can find a number of good tutorials, manuals and documentation online.  (Ebb is built using [Terra](http://terralang.org), though Ebb is designed so that programmers need not write Terra code)

## Hello, 42!

Since Tenslang doesn't support string values, let's do some arithmetic instead

```
import 'tenslang.tenslang'

local tensl getanswer()
  return 21 + 21
end

print(getanswer())
```

Save this in a file `hello42.t`.  Then, execute the command `terra hello42.t` to run the program through Terra to print out `42`.  Doing so will explode in your face.  Ow. (see below)

## Simple Ways to Get Tenslang Working

The problem we just ran into is that Terra needs to know where the tenslang [`release/`](../release) directory is before it "imports" the language.  Here are three simple ways to do that.  (note that we can't solve this problem inside of the `hello42.t` script because `import` statements are early evaluated before any of the rest of the script starts executing.

### Tenslang Launcher Script (Fast Start)

The simplest solution is to just use the provided executable launcher script.  You can find it at [`bin/tensl`](../bin/tensl).  To run `hello42.t`, we just type

```./bin/tensl hello42.t```

or any other valid path to the tensl script from wherever you happen to be.  For instance, you can add the [`bin/`](../bin) directory to your command line path to make `tensl` easily accessible wherever you find yourself in your system.

### Adding Tenslang to `TERRA_PATH`

If you work with a lot of Terra DSLs, you might find it simpler to just add tenslang's location to an environment variable.  You can do this by extending `TERRA_PATH` with `release/?.t`.  See the Terra documentation for more details.

### Custom Launcher Script

If you eventually feel like you want more control of the startup process so that you can better integrate tenslang into your tool or together with other Terra DSLs, you can always just use your own launch script and modify the `package.terrapath` variable directly there.  If you're going down this route, take a look at [`bin/tensl`](../bin/tensl) for a skeleton example.


## Using Tenslang / Terra from `C` code

For a more detailed description of ways of using Tenslang from host C code, see [`example_c/README.md`](../example_c/README.md) and example code.

There are two ways to use Tenslang from C code.  The first way is to embed a Lua/Terra interpreter in the C program.  If you go this route, then everything except for setting up the `package.terrapath` variable with the [`release/`](../release) directory is detailed in the Terra and Lua docs.

The second way is to use Tenslang scripts as code generators during your build process.  The key to this process is the `TL.compiletofile(...)` function, which is described [later on](#static-function-compilation).


# Getting Started (Writing Code)

## Tenslib
In the following we assume that the Tenslang standard library has been
included already using the line
```
local TL = require 'tenslang.tenslib'
```

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


