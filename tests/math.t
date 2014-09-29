import 'compiler.liszt'

local R = L.NewRelation(32,"relation")
R:NewField('sqrt', L.double):Load(0.0)
R:NewField('cbrt', L.double):Load(0.0)

local cbrt_test = liszt kernel(r : R)
	r.cbrt = L.cbrt(L.id(r))
end
cbrt_test(R)
R.cbrt:print()

local sqrt_test = liszt kernel (r : R)
	r.sqrt = L.sqrt(L.id(r))
end
sqrt_test(R)
R.sqrt:print()

print()

for i = 0, 32 do
	print(L.cbrt(i))
end
for i = 0, 32 do
	print(L.sqrt(i))
end