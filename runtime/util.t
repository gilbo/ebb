util = {}

function util.link_runtime ()
    local osf = assert(io.popen('uname', 'r'))
    local osname = assert(osf:read('*l'))
    osf:close()

    if osname == 'Linux' then
        terralib.linklibrary("runtime/single/libruntime_single.so")
    elseif osname == 'Darwin' then
        terralib.linklibrary("runtime/single/runtime_single.dylib")
    else
        error("Unknown Operating System")
    end
end


return util
