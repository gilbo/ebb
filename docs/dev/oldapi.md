
# Liszt API

## Sample Liszt Program

Here's a complete sample Liszt program running a heat diffusion on a 2D grid

```
import 'compiler.liszt'

local Grid = L.require('domains.grid')

local N = 5 -- 5x5 cell 2d grid
local width = 2.0 -- with size 2.0 x 2.0, and bottom-left at 0,0
local grid = Grid.NewGrid2d(N,N,{0,0},width,width)

-- load via a function...
grid.cells:NewField('temperature', L.double):Load(function(i)
  if i == 0 then return N*N else return 0 end
end)
-- load a constant
grid.cells:NewField('d_temperature', L.double):Load(0)

local K = L.NewGlobal(L.double, 1.0)

local compute_diffuse = liszt kernel ( c : grid.cells )
  if not c.is_bnd then
    var sum_diff = c( 1,0).temperature - c.temperature
                 + c(-1,0).temperature - c.temperature
                 + c(0, 1).temperature - c.temperature
                 + c(0,-1).temperature - c.temperature

    c.d_temperature = K * sum_diff
  end
end

local apply_diffuse = liszt kernel ( c : grid.cells )
  c.temperature += c.d_temperature
end

for i = 1, 1000 do
  compute_diffuse(grid.cells)
  apply_diffuse(grid.cells)
end
```


## Liszt Files

Liszt files begin with the statement
```
import 'compiler.liszt'
```
In addition to normal Lua functionality, a Liszt file will have access to a special `L` namespace from which the Liszt API can be accessed.


## `L.require` Domains

Liszt comes with support for a few domains.  (note: in the future we will expose an API for advanced users to write their own domains)  We can include a domain into our program with a statement like
```
local Grid = L.require('domains.grid')
```

## Relations

Liszt data is structured in relational tables.  Each `Relation` has some number of rows, representing elements, and some number of fields (i.e. columns) storing data associated with each element.

We can create new relations as
```
local relation = L.NewRelation(size, name)
```
for example
```
local particles = L.NewRelation(100, 'particles')
```

We can retreive the name and size of the relation

```
relation:Size()
relation:Name()
```

But most importantly, we can create new fields

## Fields

To create a new field on a relation, we call
```
relation:NewField(name, liszt_type)
```
for instance we can create a field of 3-dimensional vectors to store the positions of the particles
```
relation:NewField('position', L.vec3d)
```

You can ask a field for its type with
```
relation.field:Type()
```

As soon as we've created a field we probably want to load data into it.  A number of different options are available

### Loading Fields

The `field:Load()` function can be used in three different ways to load in initial data to a newly created field.

#### Load from constant
```
relation.field:LoadConstant(constant_value)
```
For instance, on a temperature field
```
relation.temperature:LoadConstant(0)
```
or a velocity vector
```
relation.velocity:LoadConstant({0,0,0})
```
Note that we can use a Lua list of the right length to initialize a vector-typed field, such as velocity

#### Load from list
```
relation.field:LoadList(lua_list)
```
Suppose we have a relation with `9` rows in it.  We could load the data as
```
relation.temperature:LoadList({0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1})
```

#### Load from function
````
relation.field:LoadFunction(lua_function)
```
We can also use a lua function to compute the value to set.  The function we pass in should take one argument, specifying the row index (`0` to `N-1`) and return the value for the field at that row.  For instance, if we want to dump a bunch of heat on one element and set the rest to `0` we could write
```
relation.temperature:LoadFunction(function(i)
  if i == 0 then return 1000 else return 0 end
end)
```

### Field Dumping

Once you're done with a simulation, you probably want to get the data out of the fields.  We provide two dumping functions to do this.

#### Dump to List

```
local list = relation.field:DumpToList()
```
Once dumped, the list can be written to a file or otherwise manipulated in Lua.

#### Dump Function

Alternatively, we can have a Lua function called once for each row of the field.
```
relation.field:DumpFunction(lua_function)
```
For instance, we can go through and print the temperature field of a relation
```
relation.field:DumpFunction(function(i, val)
  print(i,v)
end)
```
In the called lua funciton, the parameters are `i,val`, meaning `val` is the field value for row `i`.


## Bulk Relation Load/Save

We also provide a set of routines for saving and loading a collection of relations and their fields in bulk.

```
SaveRelationSchema
LoadRelationSchema
LoadRelationSchemaNotes
```

TODO: document this

## Types

Liszt provides the following types

```
L.float L.double L.int
L.bool
L.vector(base_type, N)
L.row(relation)
```

For convenience, we define the following vector type shorthands
```
L.vec2b L.vec3b L.vec4b
L.vec2f L.vec3f L.vec4f
L.vec2d L.vec3d L.vec4d
```
where (for instance) `L.vec3d` is shorthand for `L.vector(L.double, 3)`.

In many places, a specific relation can be used a short-hand for a specific relation.  For instance, if we want to store which cell a particle is in, we could create a field
```
particles:NewField('cell', grid.cells)
```

## Liszt Kernels

In order to compute on Liszt relations, we write Kernels

```
local kernel_name = liszt kernel ( row_name : relation )
  <body of kernel>
end
```

## Globals

You can create global variables.
```
local global_name = L.NewGlobal(type, initial_value)
```
For instance, we might want to create a variable for controlling the timestep of the simulation.
```
local dt = L.NewGlobal(L.double, 0.01)
```

TODO: change call syntax on globals and document

## `Grid` Domain

TODO: document the grid domain

## Built-in Functions

Get the length (i.e. magnitude) of a vector
```
scalar = L.length(vector)
```

Take the cross product of two vectors
```
vector = L.cross(vector_1, vector_2)
```

Take the dot product of two vectors
```
scalar = L.dot(vector_1, vector_2)
```

Print out a value for debugging
```
L.print(value)
```

Get the id for a row for the purposes of debugging
```
L.id(row)
```

assert statement for debugging
```
L.assert(test)
```








