-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"

-- Create relations and fields
local points = L.NewRelation(4, 'points')
points:NewField('x', L.int)
points:NewField('y', L.int)
local edges = L.NewRelation(4, 'edges')
edges:NewField('head', points)
edges:NewField('tail', points)


-- Create physical region
-- points._logical_region:CreatePhysicalRegion( { fields = { points.x } } )

local liszt kernel CenteredWrite(p : points)
  p.x = 2
end

CenteredWrite(points)
