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

local niltoken      = {}
local function nilconvert(val) return val==nil and niltoken or val end



-- function must return exactly one value
-- function must take at least idx_arguments
function Exports.memoize_from(idx,f)
  local cache = {}
  local function memoization_lookup(subcache, ...)
    local subval = subcache[select(1,...)]
    if select('#',...) == 1 then
      return subval, subcache
    else
      if not subval then
        subval = {}; subcache[select(1,...)] = subval
      end
      return memoization_lookup(subval, select(2,...))
    end
  end
  local function memoization_wrapper(...)
    local lookup, subcache = memoization_lookup(cache, select(idx,...))
    if not lookup then
      lookup = f(...)
      subcache[select(select('#',...),...)] = lookup
    end
    return lookup
  end
  return memoization_wrapper
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








