import 'ebb'
local L = require 'ebblib'

local R = L.NewRelation  { size = 32, name = 'R' }
R:NewField('sqrt', L.double):Load(0.0)
R:NewField('cbrt', L.double):Load(0.0)
R:NewField('sin',  L.double):Load(0.0)
R:NewField('cos',  L.double):Load(0.0)

local root_test = ebb (r : R)
	r.cbrt = L.cbrt(L.id(r))
	r.sqrt = L.sqrt(L.id(r))
end
R:foreach(root_test)

R.cbrt:print()
R.sqrt:print()

local trig_test = ebb (r : R)
	r.sin = L.sin(L.id(r))
	r.cos = L.cos(L.id(r))
end
R:foreach(trig_test)

R.sin:print()
R.cos:print()

print()
for i = 0, 32 do
	print(L.cbrt(i))
end
print()
for i = 0, 32 do
	print(L.sqrt(i))
end
print()
for i = 0, 32 do
	print(L.sin(i))
end
print()
for i = 0, 32 do
	print(L.cos(i))
end
