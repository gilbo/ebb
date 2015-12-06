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
