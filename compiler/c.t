-- location to put all C includes
-- the rest of the compiler should just terralib.require this file

-- Super ultra hack to get the directory of this file,
-- which can be mangled into the root Liszt directory.
-- Then we can make sure to stick the Liszt root on the
-- Clang path

-- Super Hacky!
local info = debug.getinfo(1, "S")
local src  = info.source
-- strip leading '@'' and trailing 'compiler/c.t'
local liszt_dir = src:sub(2,-14)


local enum_list = {
  {str='SEEK_SET',ctype='int',ttype=int},
  {str='SEEK_CUR',ctype='int',ttype=int},
}



-- If we've got CUDA available
local cuda_include = ""
if terralib.cudacompile then
  terralib.includepath = terralib.includepath..";/usr/local/cuda/include"
  cuda_include = [[
  #include "cuda_runtime.h"
  #include "driver_types.h"

  ]]
  for _,enum in ipairs({
    {str='cudaMemcpyHostToHost',ctype='int',ttype=int},
    {str='cudaMemcpyDeviceToDevice',ctype='int',ttype=int},
    {str='cudaMemcpyHostToDevice',ctype='int',ttype=int},
    {str='cudaMemcpyDeviceToHost',ctype='int',ttype=int},
  }) do
    table.insert(enum_list,enum)
  end
end




-- inject a bit of code for extracting enum values
local enum_c_define = [[
void liszt_c_terra_enum_define_helper(void **out) {
]]
for i,enum in ipairs(enum_list) do
  enum_c_define = enum_c_define ..
  '*(('..enum.ctype..'*)(out['..tostring(i-1)..'])) = '..enum.str..';\n'
end
enum_c_define = enum_c_define .. '}\n'



-- Load the blob of C defines
local c_blob = terralib.includecstring(
cuda_include ..
[[

#include "runtime/src/lmeshloader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

FILE *get_stderr () { return stderr; }

]]..
enum_c_define
)




-- now extract the enum constants
do
  -- allocate array and each entry space
  local out_arr = terralib.cast(&&opaque,
    c_blob.malloc(#enum_list * terralib.sizeof(&opaque)))
  local enum_values = {}
  for i,enum in ipairs(enum_list) do
    enum_values[i] = terralib.cast(&enum.ttype,
      c_blob.malloc(terralib.sizeof(enum.ttype)))
    out_arr[i-1] = enum_values[i]
  end

  -- execute our extraction function
  c_blob.liszt_c_terra_enum_define_helper(out_arr)

  -- now read out the values and store them in the cblob
  for i,enum in ipairs(enum_list) do
    c_blob[enum.str] = enum_values[i][0] -- [0] derefrences the pointer
    c_blob.free(enum_values[i])
  end
  c_blob.free(out_arr)
end






package.loaded["compiler.c"] = c_blob
return c_blob