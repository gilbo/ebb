import 'compiler.liszt'

local R = L.NewRelation(32,"relation")
R:NewField('sqrt', L.double):Load(0.0)
R:NewField('cbrt', L.double):Load(0.0)
R:NewField('sin',  L.double):Load(0.0)
R:NewField('cos',  L.double):Load(0.0)

local root_test = liszt kernel (r : R)
	r.cbrt = L.cbrt(L.id(r))
	r.sqrt = L.sqrt(L.id(r))
end
root_test(R)

R.cbrt:print()
R.sqrt:print()

local trig_test = liszt kernel (r : R)
	r.sin = L.sin(L.id(r))
	r.cos = L.cos(L.id(r))
end
trig_test(R)

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
