
# Liszt API

# Starting a Liszt file

## Importing the Liszt Compiler

Liszt is embedded in Lua.  Currently, this means that we need to include a line at the top of their programs to import the compiler.

```
import 'compiler.liszt'
```


## `L.default_processor`

In order to specify which processor Liszt should execute on, we can choose to set the default processor (`L.CPU` or `L.GPU`) immediately after importing the compiler.  If we omit this command entirely, Liszt will default to executing on the CPU.

```
L.default_processor = L.GPU
```


## `L.require`

Libraries and other Liszt files can be imported using Liszt's require
statement.  (The behavior mimics Lua's require statement)  For instance, if we want to import the file `trimesh.t` located in the folder `subdir` we would write
```
local TriMesh = L.require 'subdir.trimesh'
```





# Types

Liszt has the following primitive types
```
L.bool  L.float  L.double  L.int  L.uint64
```

Additionally, Liszt has `key` types.  A key-value encodes a reference to a row of some relation, making keys parameterized types
```
L.key(relation)
```
As a convenience, if we pass a relation `rel` into an API where it was expecting a type, Liszt will automatically convert that relation into `L.key(rel)`.

Lastly, both primitive types and key types can be grouped into vectors or matrices using one of the following two type constructors
```
L.vector(base_type, N)
L.matrix(base_type, Nrow, Ncol)
```

For convenience, we define aliases for common vector and matrix types.  We use the pattern `L.vec[#dimension][type]` for vectors and `L.mat[#row]x[#col][type]` for matrices.  For square matrices we also define `L.mat[N][type]`.  Some examples follow
```
L.vec3d   -- a vector of 3 doubles
L.mat4f   -- a 4x4 matrix of floats
L.mat4x4f -- another alias for a 4x4 matrix of floats
L.mat2x3i -- a 2x3 matrix of integers
L.vec3b   -- a vector of 3 booleans
```

## Type Introspection

Liszt types are Lua objects, which means they can be assigned to and passed around as Lua variables.  To help the code introspect on types when used in this way, the type objects provide a collection of methods.

-----------------------------------------------

```
L.is_type(obj)
```
returns true when the obj is a Liszt type.

-----------------------------------------------

```
typ:isScalar()
typ:isVector()
typ:isMatrix()
```
Every Liszt type is either a matrix, vector, or scalar.  If the type is a vector, we can get its size using `typ.N`.  If the type is a matrix, we can get its dimensions using `typ.Nrow` and `typ.Ncol`.

-----------------------------------------------

```
typ:baseType()
```
For vector and matrix types, the base type is the scalar type parameter.  For scalar types, this function returns the type it was called on.


### Value Types

-----------------------------------------------

```
typ:isValueType()
```
returns true when the base type is a primitive, i.e. `L.bool`, `L.float`, `L.double`, `L.int`, or `L.uint64`.

-----------------------------------------------

```
typ:isIntegral()
```
returns true when the base type is `L.int` or `L.uint64`.

-----------------------------------------------

```
typ:isNumeric()
```
returns true when the base type is `L.float`, `L.double`, `L.int`, or `L.uint64`.

-----------------------------------------------

```
typ:isLogical()
```
returns true when the base type is `L.bool`.


### Key Types

-----------------------------------------------

```
typ:isKey()
```
returns true when the base type is a key.  The relation argument to the key type can be retreived as `typ.relation`.


### Miscellaneous

-----------------------------------------------

```
typ:isPrimitive()
```
returns true when the type is a primitive. (both scalar and a value type)

-----------------------------------------------

```
typ:isScalarKey()
```
returns true when the type is a key, but not a vector of keys or matrix of keys.








# Constants

```
local c = L.Constant(typ, initial_value)
```
Liszt allows programmers to create constant values within the Lua scope.  These values must be explicitly typed.

Whenever either a Constant or a Lua value is used inside of a Liszt function, the value is substituted inline into the Liszt function as a constant.  For instance, in the following example, both `k` and `kf` are captured and inlined when `foo()` is defined.  So when `foo()` is mapped over the cells, both lines compute `c.f += 23 * c.g` even though the value of the Lua variable `k` has later been changed.

```
local kf = L.Constant(L.float, 23)
local k  = 23
local liszt foo ( c : cells )
  c.f += kf * c.g
  c.f += k * c.g
end

k = 11

cells:map(foo)
```

Constants provide us with two benefits.  1) Constants allow us to explicitly type constant values. (we do not know for sure what value k will be assigned above) 2) Constants allow us to explicitly document values that we intend as constants, so that our code is more easily understandable to other programmers.

-------------------

```
L.is_constant(obj)
```
Test whether a Lua object is a constant or not

-------------------

```
constant:get()
```
Retreive the Lua value of the constant


# Globals

```
local g = L.Global(typ, initial_value)
```
Liszt allows programmers to create 'global' variables, i.e. data that is shared across all relations, rows of relations, and functions mapped over those relations.

-------------------

```
L.is_global(obj)
```
Test whether a Lua object is a global or not.

-------------------

```
global:set(val)
```

Set the value of a global variable from Lua code.

-------------------

```
global:get()
```

Get the value of a global variable from Lua code.  If you're performance tuning, be aware that `get()` is a blocking call.  If necessary, the main control thread will hang while waiting for the newest value of the global.









# Relations

```
local rel = L.NewRelation({
  name = 'rel',
  mode = ['PLAIN' | 'GRID' | 'ELASTIC'],
 [size = n_rows,]           -- required if mode ~= 'GRID'
 [dims = {nx,ny,...},]      -- required if mode == 'GRID'
 [periodic = {px,py,...},]  -- optional if mode == 'GRID'
})
```

Liszt relations are created by passing a table of named arguments to the `NewRelation` constructor.  Descriptions of each argument follow

- `name` assigns a name to the relation for debugging.  We recommend setting this to be the same as the name of the variable you assign the relation to.
- `mode` a relation can exist in one of 4 modes, and can be created initially in 3 of those 4 modes.  If this argument is omitted the mode is inferred as `'PLAIN'` if a `size` argument is provided, or `'GRID'` if a `dims` argument is provided.
- `size` specifies the number of rows in a `'PLAIN'` or `'ELASTIC'` relation.
- `dims` specifies a list of dimension sizes for a `'GRID'` relation.  Either two numbers can be provided (a 2d grid) or three (a 3d grid).  Other numbers of dimensions are not supported.
- `periodic` is an optional argument when creating a `'GRID'` relation.  It expects a liszt parallel to `dims` with boolean entries.  If the entry is true for a given dimension, then Liszt will wrap around all indices in that dimension.  (e.g. consider indexing a row of a 2d relation created with `periodic={true,false}` and `dims={10,12}`.  The key `[13,4]` will be automatically wrapped around into key `[3,4]`, but trying to access data at key `[4,13]` will cause an error) 

------------------

As mentioned, a relation in Liszt can be in one of 4 different states, never more than one at a time.

```
Plain    Grouped    Grid    Elastic
```

Not all functionality is available in every relation mode.

------------------

```
L.is_relation(obj)
```

Test whether a Lua object is a relation or not.

------------------

```
local str = rel:Name()
```

Retrieve the name of a relation.

------------------

```
local n = rel:Size()
```

Retrieve the number of rows in a relation.  If the relation is in Grid mode, then this is the product of the dimensions.

------------------

```
rel:print()
```

print the contents of this relation's fields to the console (for debugging).


## Plain Relations

```
rel:isPlain()
```

Test whether a relation is in Plain mode.


## Grouped Relations

Plain mode relations can be grouped, putting them into Grouped mode.  Elastic and Grid mode relations cannot be grouped.

```
rel:GroupBy(field)
```

`GroupBy()` expects either a field defined on relation `rel` or the name (i.e. a string) of a field defined on `rel` as its argument.  This field should have type `L.key(src_rel)`, and should already be sorted by the key values in this field.  (We plan to eventually support automatically sorting this data, but haven't yet done so.)

```
for dest_key in L.Where(grouped_field, source_key) do
  ...
end
```

Once a relation `dest_rel` has been grouped by a field `grouped_field` of type `L.key(source_rel)`, we can find all the keys of `dest_rel` where `dest_key.grouped_field == source_key` given a key from `source_rel`.  We do this using a query loop, as shown above.

For example, we could group edges by their tail vertex, and then take a neighborhood average, using a query loop.

```
edges:GroupBy('tail')

local liszt foo(v : vertices)
  var sum = 0
  for e in L.Where(edges.tail, v) do
    sum += e.head.f
  end
  v.f_sum = sum
end
```

For convenience, Liszt allows us to compose key field accesses with `L.Where()` queries.  For instance, we could rewrite the above loop instead as

```
  for nv in L.Where(edges.tail, v).head do
    sum += nv.f
  end
```

In practice, these query loops are hidden behind macros so that a programmer writes something more like

```
  for nv in v.neighbors do
    sum += nv.f
  end
```

------------------

```
rel:isGrouped()
```

Test whether a relation is in Grouped mode

------------------

```
rel:GroupedKeyField()
```

Return the field that `rel` is currently grouped by.  If `rel` is not grouped, then this call returns `nil`.


## Elastic Relations

```
rel:isElastic()
```

Test whether a relation is in Elastic mode.

------------------

```
rel:isFragmented()
```

Test whether a relation is fragmented.  Many functions will fail if called on a fragmented relation or a fragmented relation's fields. (e.g. I/O related functions)  Only Elastic mode relations can be fragmented.

------------------

```
rel:Defrag()
```

When called on a fragmented Elastic mode relation, `Defrag()` will ensure that the rows of the relation are densely packed in memory.


## Grid Relations

Grid mode relations support a special way of accessing neighboring elements apart from fields of keys.  This functionality is provided by the `L.Affine` topological join.

```
local dest_key = L.Affine(destination_grid, affine_matrix, source_key)
```

`L.Affine` can only be called inside of a Liszt function.  It transforms the `source_key` into a key of type `L.key(destination_grid)`.  Specifically, if `[xid,yid]` or `[xid,yid,zid]` are the indices of the `source_key`, then the indices of the resulting key are produced by applying the affine transformation matrix.  If the source grid has `m` dimensions and the destination grid has `n` dimensions, then the affine transformation matrix should have `n` rows and `m+1` columns.  The last column of the matrix encodes the translational offset, so that the transformation computation consists of multiplying by the first `nxm` block and then adding the last `nx1` column.  Finally, the supplied `affine_matrix` must be a literal, not a variable matrix value.

Some example uses of `L.Affine` follow.

```
local liszt copy_from_right( c : cells )
  c.f_copy = L.Affine(cells, {{1,0,1},
                              {0,1,0}}, c).f
end
```
This example copies data over from the right neighbor in a 2d grid.

```
local liszt cell_to_vertex( c : cells )
  var v = L.Affine(vertices, {{1,0,0},
                              {0,1,0}}, c)
end
```
This example produces the vertex with the same indices as the cell, and does nothing in particular with it.

```
local liszt up_step( cc : coarse_cells )
  var fc = L.Affine(fine_cells, {{2,0,0},
                                 {0,2,0}}, cc)
  var sum = fc.f
          + L.Affine(fine_cells, {{1,0,1},
                                  {0,1,0}}, fc).f
          + L.Affine(fine_cells, {{1,0,0},
                                  {0,1,1}}, fc).f
          + L.Affine(fine_cells, {{1,0,1},
                                  {0,1,1}}, fc).f
  cc.f = sum / 4.0
end
```
This example performs an integration of a field stored at 4 fine cells into the corresponding coarse cell. (e.g. in a multi-grid scheme) In practice, these `L.Affine` calls would probably be hidden behind macros, producing code more like the following

```
local liszt up_step( cc : coarse_cells )
  var fc = cc.fine
  cc.f = ( fc.f + fc(1,0).f + fc(0,1).f + fc(1,1).f ) / 4.0
end
```

------------------

```
rel:isGrid()
```

Test whether a relation is in Grid mode.

------------------

```
rel:Dims()
```

Retrieve a copy of the `dims` list used to create this grid relation.  If called on a relation not in Grid mode, then `{ rel:Size() }` is returned instead.

------------------

```
rel:nDims()
```

Return `2` or `3` identifying how many dimensions this grid has.  If called on a relation not in Grid mode, then `1` will be returned

------------------

```
rel:Periodicity()
```

Retreive a copy of the `periodic` list specified when creating this grid relation.  If called on a relation not in Grid mode, then `{ false }` is returned instead.


## Subsets

So long as a relation is not in Elastic mode, we can define subsets of the relation.

```
rel:NewSubsetFromFunction(name, lua_function)
```

Define a new subset of `rel` with name `name` defined by `lua_function` as follows.  `lua_function` is called once for each row and is expected to either return true (include this row in the subset) or false (exclude it); `lua_function()` is given arguments `lua_function(id, [yid, [zid]])` depending on how many dimensions `rel` has.

NOTE: id conventions for defining subsets are currently subject to change; we expect the whole I/O system to change dramatically as Liszt incorporates support for distributed runtimes.

Once defined, a subset can be accessed as `rel.subset_name`.

------------------

```
rel:hasSubsets()
```

Test whether a relation has subsets defined on it.

------------------

```
L.is_subset(obj)
```

Test whether a Lua object is a Liszt subset.

------------------

```
subset:Relation()
```

return the relation that `subset` is a subset of.

------------------

```
subset:Name()
```

return the name of the subset.

------------------

```
subset:FullName()
```

return the name of the full relation and subset concatenated: `"rel_name.subset_name"`






# Fields

```
rel:NewField(name, type)
```

Create a field on a relation with the given `name` and `type`.  The field will now be accessible via `rel.name`.  (as a convenience, `rel:NewField()` returns the newly created field, allowing for method chaining, as in `rel:NewField(...):Load(...)`)

------------------

```
L.is_field(obj)
```

Test whether a Lua object is a field or not.

------------------

```
field:Name()
```

return the name of the field.

------------------

```
field:FullName()
```

return the name of the relation and field concatenated: `"rel_name.field_name"`

------------------

```
field:Relation()
```

return the relation that this field is defined on.

------------------

```
field:Size()
```

return the size of the relation that this field is defined on.

------------------

```
field:Type()
```

return the type of this field.

------------------

```
field:print()
```

print the contents of this field to the console (for debugging).

------------------

```
rel:Copy({from = f1, to = f2})
```

copy the contents of field `f1` into field `f2` (assuming both are fields of relation `rel`).  `f1` and `f2` may be supplied as either field objects or strings identifying names of fields.

------------------

```
rel:Swap(f1,f2)
```

swap the contents of field `f1` into field `f2` (assuming both are fields of relation `rel`).  `f1` and `f2` may be supplied as either field objects or strings identifying names of fields.




## Field Input and Output (Load and Dump)

`Load()` is used to initialize field data.  It can be used in 3 different ways.

```
field:Load(constant)
field:Load(lua_list)
field:Load(lua_function)
```

The first variant assigns the value `constant` for every row of the relation.  The second variant takes a Lua list of values as its argument; the list must have the right number of entries `#lua_list == field:Size()`.  The last form takes a function that is called once for each row and is expected to return the value to initialize that row's entry to; `lua_function()` is given arguments `lua_function(id, [yid, [zid]])`.

NOTE: id conventions for loading are currently subject to change; we expect the whole I/O system to change dramatically as Liszt incorporates support for distributed runtimes.

As an example, the following function initializes every row except the first row with temperature `0`.  The first row gets temperature `1000`.

```
vertices.temperature:Load(function(i)
  if i == 0 then return 1000 else return 0 end
end)
```

------------------

```
field:DumpToList()
```

Opposite of loading via a list; this function returns the contents of `field` converted into a Lua list value.

------------------

```
field:DumpFunction(lua_function)
```

The argument function `lua_function` is called once for each row, and passed arguments `lua_function(value, id, [yid, [zid]])` depending on how many dimensions `field:Relation()` has.

NOTE: id conventions for dumping are currently subject to change; we expect the whole I/O system to change dramatically as Liszt incorporates support for distributed runtimes.

As an example, consider dumping velocity from a grid and printing it to the console.

```
cells.velocity:DumpFunction(function(v, xid, yid)
  print('cell at ', xid, yid, ' has velocity ', v[1], v[2])
end)
```

------------------

Sometimes, we'd like to be able to dump multiple fields from a relation at the same time.

```
rel:DumpJoint(fields_list, lua_function)
```

`DumpJoint()` calls `lua_function` once for each row of `rel`, with row ids and values for all of the requested fields.  These arguments are supplied in the order `lua_function(ids, [field1_value, field2_value, ...])`.  The argument `ids` is a Lua list containing between 1 and 3 values depending on the number of dimensions `rel` has.  `fields_list` is a list of either field objects or strings identifying particular fields.  The values passed to `lua_function` are passed in the same order as the fields are listed in `fields_list`.

NOTE: id conventions for dumping are currently subject to change; we expect the whole I/O system to change dramatically as Liszt incorporates support for distributed runtimes.

As an example, suppose we want to print both the velocity and pressure of a grid to the console with one line for each row.

```
cells:DumpJoint({'velocity', 'pressure'},
  function(ids, v, p)
    print(ids[1], ids[2], ': ', v[1], v[2], p)
  end)
```


# Data Layout Descriptors (DLDs)

In order to support interoperability with external libraries (especially linear solvers) Liszt provides a way of requesting direct access to its underlying data storage.  Data Layout Descriptors (DLDs) are meta-data describing this data layout.

However, client code and external libraries do not have any way to control how Liszt arranges its internal representations.  As such, external code is required to either marshall/copy data into its own storage, or to adapt to however Liszt chooses to arrange memory.

```
field:getDLD()
```

This call returns a DLD describing the data layout of a field.

NOTE: The exact format of DLDs is preliminary and subject to change.  The following describes the current convention.

- `dld.location` A string describing the location of the data (either `'CPU'` or `'GPU'`)

- `dld.type.dimensions` if the type is a scalar, this is an empty Lua list `{}`; if the type is a vector this is a Lua list containing one number, the length of the vector `{N}`; if the type is a matrix, this is a Lua list containing the dimensions `{Nrow,Ncol}`.

- `dld.type.base_type_str` a string describing the C type of the field's base type

- `dld.type.base_bytes` a number giving the size of the base type in bytes

- `dld.logical_size` describes the number of items in the array, i.e. the number of rows

- `dld.address` a pointer to the beginning of the data

- `dld.stride` number, how many bytes to advance the pointer in order to find the next row's data







# Macros

Liszt supports macros in the style of Terra as an advanced feature.

```
local m = L.NewMacro(lua_generator_function)
```

`NewMacro` takes a Lua generator function as its argument.  This function is passed Liszt ASTs as arguments and expected to return a Liszt AST as its result.  For instance, the following macro implements addition.

```
local sum_macro = L.NewMacro(function(arg1, arg2)
  return liszt `arg1 + arg2
end)
```

----------------

```
L.is_macro(obj)
```

Test whether a Lua object is a macro or not.


## Field Macros

```
rel:NewFieldMacro(name, macro)
```

Install `macro` with name `name` on relation `rel` as if it were a function.  Inside Liszt code, whenever a key from `rel` accesses `name`, the macro will be expanded and passed the key as its only argument.  A macro bound to the special name `__apply_macro` will instead be expanded wherever a programmer calls a key from the relation, and passed the key along with any arguments as arguments to the macro function.

This feature allows us to expose more friendly syntax to the users of geometric domain libraries.

```
vertices:NewFieldMacro('edges', L.NewMacro(function(v)
  return liszt `L.Where(edges.tail, v)
end))
```
This macro now allows client programmers to write `for e in v.edges do ... end` when `v` is a key from `vertices`.

```
cells:NewFieldMacro('__apply_macro', L.NewMacro(function(c, xoff, yoff)
  return liszt `L.Affine(cells, {{1, 0, xoff},
                                 {0, 1, yoff}}, c)
end))
```
This macro now allows client programmers to write `c(-1,2)` in order to access the neighbor of `c` that's `-1` away in the x direction and `2` away in the y direction.









# Functions

Liszt functions can be declared similarly to Lua functions, except the keyword `function` is replaced by `liszt`.

```
liszt foo() ... end
local liszt foo() ... end
local foo = liszt() ... end
local namespace = {}
liszt namespace.foo() ... end
```

Arguments to Liszt functions can be optionally typed.

```
local liszt foo( a : L.vec3d, b : L.float ) ... end
local liszt bar( a : L.vec3d, b ) ... end
local liszt baz( a, b ) ... end
local liszt bop( c : cells ) ... end
local liszt fim( c )
```

When the Lua interpreter encounters the definition of a Liszt function, all Lua values used in the definition of the Liszt function are captured.

Then, when the function is used for the first time, Liszt will type-check the function.  If the function is never used, but contains type-errors, none will be reported.  Untyped functions are implicitly polymorphic.  Liszt will create, typecheck and compile a separate version of each polymorphic function for each different type signature it's called with.

----------------

```
L.is_function(obj)
```

Test whether a Lua object is a Liszt function or not.

----------------

## Field Functions

```
rel:NewFieldFunction(name, liszt_function)
```

Liszt functions can be installed on relations as if they were fields.  Liszt expects `liszt_function` to have exactly one argument of type `L.key(rel)`, i.e. `liszt_function( r : rel )`.  Then, whenever a key from `rel` is accessed as `r.name`, that access will be replaced the call `liszt_function(r)`.  In this way, it's possible for Liszt programmers to easily replace fields of data with dynamically computed fields or vice-versa.



# Built-in Functions

Liszt provides a number of built-in functions.

```
L.cos
L.sin
L.tan
L.acos
L.asin
L.atan
L.fmod
L.pow
L.sqrt
L.cbrt
L.floor
L.ceil
L.fabs
```

These standard mathematical functions are provided as built-ins.  They will be replaced with GPU/CUDA intrinsics when code is run on the GPU, unlike the standard C math library.

----------------

```
L.cross(a,b)
L.dot(a,b)
L.length(a)
```

These functions work on Liszt vector values and are provided for convenience.

----------------

```
L.id(k)

L.xid(k)
L.yid(k)
L.zid(k)
```

These functions are provided to extract key identifiers. (`L.uint64` values)  If `k` is from a Grid mode relation, then the `xid`/`yid`/`zid` variants should be used; otherwise the `id` variant should be used.

NOTE: the values returned from `L.id()` are likely to change when Liszt starts supporting distributed runtimes.

----------------

```
L.assert( boolean_expression )
```

Terminate execution if the boolean expression evaluates to false

----------------

```
L.print(value1, [value2, ...])
```

Print some number of values to the console for debugging.  No ordering guarantees are provided.  Furthermore, since Liszt does not support string data, no formatting is currently allowed.












