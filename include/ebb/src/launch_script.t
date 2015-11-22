
-- This file is really dumb.
-- Its entire goal is to provide an
-- opportunity to shim in extra Lua
-- junk before we launch an Ebb program.


-- Here, we wrap the call with error handling.
-- So now, we'll always get a stack trace with any errors
-- SUH-WEET!

local return_code = 0

local function top_level_err_handler ( errobj )
  local err = tostring(errobj)
  if not string.match(err, 'stack traceback:') then
    err = err .. '\n' .. debug.traceback()
  end
  print(err)
  os.exit(1)
end

script_filename = arg[0]

exit_code = xpcall( function ()
  assert(terralib.loadfile(script_filename))()

  if terralib.cudacompile then
    -- make sure all CUDA computations have finished
    -- before we exit
    local errcode = require('ebb.src.gpu_util').device_sync()
    if errcode ~= 0 then
      error('saw CUDA error code when exiting: '..errcode)
    end
  end

end, top_level_err_handler)
