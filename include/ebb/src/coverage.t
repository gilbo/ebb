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

if not rawget(_G,'COVERAGE_MODULE_HAS_BEEN_INSTALLED') then
  local ffi = require 'ffi'

  _G['COVERAGE_MODULE_HAS_BEEN_INSTALLED'] = true
  local coverageloader, err = loadfile("coverageinfo.lua")
  --print('FOUND COVERAGE ', coverageloader, err)

  local filetable = coverageloader and coverageloader() or {}
  local function dumplineinfo()
    local F = io.open("coverageinfo.lua","w")
    F:write("return {\n")
    for filename,linetable in pairs(filetable) do
      F:write("['"..filename.."']={\n")
      for linenum,count in pairs(linetable) do
        F:write("["..linenum.."]="..count..",\n")
      end
      F:write("},\n")
    end
    F:write("}\n")
    F:close()
  end
  local function debughook(event)
    local info = debug.getinfo(2,"Sl")
    -- exclude for instance, metaprogrammed lua code
    if string.sub(info.source, 1,1) == '@' then
      local linetable = filetable[info.source]
      if not linetable then
        linetable = {}
        filetable[info.source] = linetable
      end
      linetable[info.currentline] = (linetable[info.currentline] or 0) + 1
    end
  end
  debug.sethook(debughook,"l")
  -- make a fake ffi object that causes dumplineinfo to be called when
  -- the lua state is removed
  ffi.cdef [[
    typedef struct {} __linecoverage;
  ]]
  ffi.metatype("__linecoverage", { __gc = dumplineinfo } )
  _G[{}] = ffi.new("__linecoverage")
end
