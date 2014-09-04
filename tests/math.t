import 'compiler.liszt'

local R = L.NewRelation(32,"relation")

local cbrt_test = liszt kernel(r : R)
	L.print(L.cbrt(L.id(r)))
end
cbrt_test(R)

print()

for i = 0, 32 do
	print(L.cbrt(i))
end