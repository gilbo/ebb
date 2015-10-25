
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
