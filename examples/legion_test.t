-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"

-- Declaring new relation
local points = L.NewRelation(5, 'points')
