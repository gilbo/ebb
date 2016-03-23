-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.


-- file/module namespace table
local Exports = {}
package.loaded["ebb.src.util"] = Exports

local C             = require "ebb.src.c"

local lookuptoken   = {}
local niltoken      = {}
local function nilconvert(val) return val==nil and niltoken or val end



-- function must return exactly one value
-- function must take at least idx_arguments
function Exports.memoize_from(idx,f)
  local cache = {}
  local function memoization_wrapper(...)
    local args     = {select(idx,...)}
    local subcache = cache
    for i=1,#args do
      local sub = subcache[nilconvert(args[i])]
      if not sub then
        sub = {}
        subcache[nilconvert(args[i])] = sub
      end
      subcache = sub
    end
    local lookup = subcache[lookuptoken]
    if not lookup then
      lookup = f(...)
      subcache[lookuptoken] = lookup
    end
    return lookup
  end
  return memoization_wrapper
end
function Exports.memoize(f)
  return Exports.memoize_from(1,f)
end

-- function must return exactly one value
-- function must take a single table of named arguments
-- named arguments (potentially) used in memoization are supplied
-- in the list 'keys'
function Exports.memoize_named(keys, f)
  local cache = {}
  local function named_memoization_wrapper(args)
    local subcache = cache
    for i=1,#keys-1 do
      local keyval  = nilconvert( args[keys[i]] )
      local lookup  = subcache[keyval]
      if not lookup then
        lookup = {}; subcache[keyval] = lookup
      end
      subcache = lookup
    end
    local keyval  = nilconvert( args[keys[#keys]] )
    local val     = subcache[keyval]
    if not val then
      val = f(args); subcache[keyval] = val
    end
    return val
  end
  return named_memoization_wrapper
end

-------------------------------------------------------------------------------

local Cache   = {}
Cache.__index = Cache

local function cache_basic_get(subcache, k, ...)
  local args = {...}
  for i=1,k-1 do
    local key = nilconvert(args[i])
    local val = subcache[key]
    if not val then
      val = {}
      subcache[key] = val
    end
    subcache = val
  end
  return nilconvert(args[k]), subcache
end

local function cache_insert_entry(cache, val, ...) -- ... = keys
  local key, subcache = cache_basic_get(cache._cache, cache._n_args, ...)
  subcache[key] = val
end

local function cache_delete_entry(cache, ...) -- ... = keys
  local key, subcache = cache_basic_get(cache._cache, cache._n_args, ...)
  subcache[key] = nil
end

local function cache_lookup(cache, ...)
  local key, subcache = cache_basic_get(cache._cache, cache._n_args, ...)
  return subcache[key]
end

function Exports.new_named_cache(names)
  local function unpack_args(args)
    local keys = {}
    for i,n in ipairs(names) do keys[i] = args[n] end
    return unpack(keys)
  end
  return setmetatable({
    _cache = {},
    _n_args = #names,
    insert = function(self, val, args)
      return cache_insert_entry(self, val, unpack_args(args))
    end,
    delete = function(args)
      return cache_delete_entry(self, unpack_args(args))
    end,
    lookup = function(args)
      return cache_lookup(self, unpack_args(args))
    end,
  }, Cache)
end













