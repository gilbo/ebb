util = {}

-- Super Hacky!
local info = debug.getinfo(1, "S")
local src  = info.source
-- strip leading '@'' and trailing 'util.t'
local runtime_dir = src:sub(2,-8)

function util.link_runtime ()
    terralib.linklibrary(runtime_dir.."/libsingle_runtime.so")
end

return util
