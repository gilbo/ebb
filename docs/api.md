
# Liszt API

## A Liszt file



```
import 'compiler.liszt'

local Grid = L.

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
local function init_temp (i)
  if i == 0 then
    return 1000
  else
    return 0
  end
end

M.vertices:NewField('flux',        L.float):LoadConstant(0)
M.vertices:NewField('jacobistep',  L.float):LoadConstant(0)
M.vertices:NewField('temperature', L.float):LoadFunction(init_temp)

local compute_step = liszt_kernel(e : M.edges)
  var v1   = e.head
  var v2   = e.tail
  var dp   = v1.position - v2.position
  var dt   = v1.temperature - v2.temperature
  var step = 1.0 / L.length(dp)

  v1.flux -= dt * step
  v2.flux += dt * step

  v1.jacobistep += step
  v2.jacobistep += step
end

local propagate_temp = liszt_kernel (p : M.vertices)
  p.temperature += L.float(.01) * p.flux / p.jacobistep
end

local clear = liszt_kernel (p : M.vertices)
  p.flux       = 0
  p.jacobistep = 0
end

for i = 1, 1000 do
  compute_step(M.edges)
  propagate_temp(M.vertices)
  clear(M.vertices)
end

M.vertices.temperature:print()
```


## The `L` namespace



## Types



float float
uint64  uint64
bool  bool
uint8 uint8
double  double
vector  function: 0x04d6aa68
row function: 0x04d73ed8
int int
uint  uint

****

vec2b Vector(bool,2)
vec3b Vector(bool,3)
vec3d Vector(double,3)
vec4f Vector(float,4)
vec4b Vector(bool,4)
vec4d Vector(double,4)
vec2d Vector(double,2)
vec3f Vector(float,3)
vec2f Vector(float,2)

addr  uint64
internal  function: 0x04d73f88
error error


singleCore  Single Core Runtime
gpu GPU Runtime


is_vector function: 0x04c093e0
is_function function: 0x04c164d8
is_macro  function: 0x04c08cd0
is_global function: 0x04c01178
is_relation function: 0x04c09ac8
is_field  function: 0x04c097a0
LMacro  table: 0x04c08c48
LGlobal table: 0x04c09410
LKernel table: 0x04c08d10
LField  table: 0x04c10310
LRelation table: 0x04c09718
LVector table: 0x04c11600

NewVector function: 0x04c09768
NewKernel function: 0x04e7a538
is_kernel function: 0x04c08d98


NewMacro  function: 0x04b75f50
id  table: 0x04cf2a70
UNSAFE_ROW  table: 0x04cf2bc0
length  table: 0x04d16370
all table: 0x04d165f8
cross table: 0x04d021f0
dot table: 0x04d02178
print table: 0x04d02038
any table: 0x04d16b58
assert  table: 0x04cf25d0

NewGlobal function: 0x04b79478
NewRelation function: 0x04e03d00

SaveRelationSchema  function: 0x04e21dc8
LoadRelationSchema  function: 0x04e21f58
LoadRelationSchemaNotes function: 0x04e21e78


record  function: 0x04d6aaf8
query function: 0x04d73ff0
Where table: 0x04b75e20






