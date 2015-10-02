import "ebb"
local Grid  = require 'ebb.domains.grid'

-- parameters
local N = 140
local ITERS = 50

local TEST_MODE = true
local LE = rawget(_G, '_legion_env')
local USE_TIMERS = ((LE == nil) and not TEST_MODE)

-- timer
local Timer = {}
local timer_init, timer_exec

if USE_TIMERS then
Timer.__index = Timer

function Timer.New()
  local timer = { start = 0, finish = 0, total = 0 }
  setmetatable(timer, Timer)
  return timer
end

function Timer:Reset()
  self.start = 0
  self.finish = 0
  self.total = 0
end

function Timer:Start()
  self.start = terralib.currenttimeinseconds()
end

function Timer:Stop()
  self.finish = terralib.currenttimeinseconds()
  self.total = self.total + self.finish - self.start
  self.start = 0
end

function Timer:GetTime()
  return self.total
end

timer_init  = Timer.New()
timer_exec  = Timer.New()
end

-- grid relation
local grid = Grid.NewGrid3d {
    size   = { N,    N,    N  },
    origin = {-N/2, -N/2, -N/2},
    width  = { N,    N  ,  N},
    boundary_depth    = { 1  ,  1  ,  1  },
    periodic_boundary = {true, true, true}
}

if USE_TIMERS then
timer_init:Start()
end

-- fields
grid.vertices:NewField('x', L.vec3f):Load({2, 2, 2})
grid.cells:NewField('x', L.vec3f):Load({10, 10, 10})
grid.vertices:NewField('y', L.vec3f):Load({0, 0, 0})
grid.cells:NewField('y', L.vec3f):Load({0, 0, 0})

if USE_TIMERS then
timer_init:Stop()
timer_exec:Start()
end

-- empty kernel
local ebb empty(c)
end

-- scatter field 'field' from 'from' relation to 'to' relation
local ebb Scatter(from, to, field)
  from[to](-1,-1,-1)[field] += from[field]/L.float(8)
  from[to](-1,-1, 1)[field] += from[field]/L.float(8)
  from[to](-1, 1,-1)[field] += from[field]/L.float(8)
  from[to](-1, 1, 1)[field] += from[field]/L.float(8)
  from[to]( 1,-1,-1)[field] += from[field]/L.float(8)
  from[to]( 1,-1, 1)[field] += from[field]/L.float(8)
  from[to]( 1, 1,-1)[field] += from[field]/L.float(8)
  from[to]( 1, 1, 1)[field] += from[field]/L.float(8)
end

-- copy field x to y, to test for task parallelism
local ebb Update(rel, field1, field2)
  rel[field2] = (L.float(1.2)*rel(-1, -1, -1)[field1] +
                 L.float(2.7)*rel(1, 1, 1)[field1])/L.float(3.1)
end

-- loop
for i = 1, ITERS do
  -- grid.cells:foreach(empty)
  grid.cells:foreach(Scatter, 'vertex', 'x')
  grid.cells:foreach(Update, 'x', 'y')
  grid.vertices:foreach(Scatter, 'cell', 'x')
  grid.vertices:foreach(Scatter, 'cell', 'y')
end

if USE_TIMERS then
timer_exec:Stop()
print("Total initialization time = " .. timer_init:GetTime() .. " seconds")
print("Total compile + exec time = " .. timer_exec:GetTime() .. " seconds")
end

if not TEST_MODE then
  Scatter:getCompileTime():print()
  Scatter:getExecutionTime():print()
end
