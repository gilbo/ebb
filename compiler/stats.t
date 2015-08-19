
local Stats = {}
package.loaded["compiler.stats"] = Stats


-- NOTE: I'm currently measuring 1 us overhead on wrapping a simple
--       timer around function launches; this seems negligible to me,
--       so I'm gonna leave all stats enabled by default
-- NOTE: I can shave off half of that microsecond by flipping this
--        variable, but to remove all the overhead the actual inline timer
--        start/stop statements need to be removed from the code
local STATS_ON = true

------------------------------------------------------------------------------
--[[ Counters                                                             ]]--
------------------------------------------------------------------------------

local Counter     = {}
Counter.__index   = Counter

function Stats.NewCounter(name)
  local counter = setmetatable({
    _val  = 0,
    _name = tostring(name),
  }, Counter)
  return counter
end

function Counter:increment()
  self._val = self._val + 1
end

function Counter:get()
  return self._val
end
function Counter:print(prefix)
  prefix = (prefix or self._name) .. ': '
  print(prefix, self._val)
end


------------------------------------------------------------------------------
--[[ Timers                                                               ]]--
------------------------------------------------------------------------------

local Timer     = {}
Timer.__index   = Timer

function Stats.NewTimer(name)
  local timer = setmetatable({
    _name   = tostring(name),
    _start  = nil,

    _count  = 0,
    _min_ms = math.huge,
    _max_ms = 0,
    _sum_ms = 0,
  }, Timer)
  return timer
end

function Timer:setName(name)
  self._name = tostring(name)
end
-- prefix should be supplied if printed
function Timer.__add(lhs, rhs)
  if getmetatable(lhs) ~= Timer or getmetatable(rhs) ~= Timer then
    error('cannot add a Timer to a non-Timer')
  end
  local sumtimer = setmetatable({
    _name   = '',
    _start  = nil,

    _count  = lhs._count + rhs._count,
    _min_ms = math.min(lhs._min_ms, rhs._min_ms),
    _max_ms = math.max(lhs._max_ms, rhs._max_ms),
    _sum_ms = lhs._sum_ms + rhs._sum_ms,
  }, Timer)
  return sumtimer
end

function Timer:start(timestamp_in_ms)
  if not timestamp_in_ms then
    timestamp_in_ms = terralib.currenttimeinseconds() * 1.0e3
  end
  self._start = timestamp_in_ms
  return timestamp_in_ms
end
function Timer:stop(timestamp_in_ms)
  if not timestamp_in_ms then
    timestamp_in_ms = terralib.currenttimeinseconds() * 1.0e3
  end
  if not self._start then error('must match timer stops with starts') end
  local dt = timestamp_in_ms - self._start
  self._start = nil

  self._count   = self._count + 1
  self._min_ms  = math.min(self._min_ms, dt)
  self._max_ms  = math.max(self._max_ms, dt)
  self._sum_ms  = self._sum_ms + dt

  return timestamp_in_ms
end

-- Bulk routines for convenience
function Stats.StopAndStart(stop_counters, start_counters)
  local timestamp_in_ms = terralib.currenttimeinseconds() * 1.0e3
  for _,timer in ipairs(stop_counters) do
    timer:stop(timestamp_in_ms)
  end
  for _,timer in ipairs(start_counters) do
    timer:start(timestamp_in_ms)
  end
  return timestamp_in_ms
end
function Stats.Start(counters)
  return Stats.StopAndStart({},counters)
end
function Stats.Stop(counters)
  return Stats.StopAndStart(counters,{})
end

if not STATS_ON then
function Timer:start() end
function Timer:stop() end
function Stats.StopAndStart() end
end


function Timer:getcount()     return self._count    end
function Timer:getmin()       return self._min_ms   end
function Timer:getmax()       return self._max_ms   end
function Timer:getsum()       return self._sum_ms   end
function Timer:getavg()       return self._sum_ms / self._count   end

function Timer:print(prefix)
  prefix = (prefix or self._name) .. ':'
  print(prefix)
  print('  avg:   '..tostring(self:getavg()))
  print('  min:   '..tostring(self._min_ms))
  print('  max:   '..tostring(self._max_ms))
  print('  sum:   '..tostring(self._sum_ms))
  print('  count: '..tostring(self._count))
end



------------------------------------------------------------------------------
--[[ Global Statistics                                                    ]]--
------------------------------------------------------------------------------

local global_stat_table = {}

function Stats.NewGlobalCounter(name)
  name = tostring(name)
  if global_stat_table[name] then
    error("stat name '"..name.."' is already being used") end
  local counter = Stats.NewCounter(name)
  global_stat_table[name] = counter
  return counter
end

function Stats.NewGlobalTimer(name)
  name = tostring(name)
  if global_stat_table[name] then
    error("stat name '"..name.."' is already being used") end
  local timer = Stats.NewTimer(name)
  global_stat_table[name] = timer
  return timer
end

-- get statistic
function Stats.GetGlobalStat(name)
  local lookup = global_stat_table[name]
  if not lookup then error("could not find global stat '"..name.."'") end
  return lookup
end

--[[
function Stats.Enable()
  STATS_ON = true
end

function Stats.Disable()
  STATS_ON = false
end
]]






