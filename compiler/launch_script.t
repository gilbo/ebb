
-- This file is really dumb.
-- Its entire goal is to provide an
-- opportunity to shim in extra Lua
-- junk before we launch a Liszt program.


-- Here, we wrap the call with error handling.
-- So now, we'll always get a stack trace with any errors
-- SUH-WEET!

local return_code = 0

local function top_level_err_handler ( errobj )
  local err = tostring(errobj)
  if string.match(err, 'stack traceback:') then
    print(err)
  else
    print(err .. '\n' .. debug.traceback())
  end
  os.exit(1)
end

script_filename = arg[1]

success = xpcall( function ()
  assert(terralib.loadfile(script_filename))()
end, top_level_err_handler)

if success then
  os.exit(0)
else
  os.exit(1)
end