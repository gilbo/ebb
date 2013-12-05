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

return terralib.includecstring([[
#include "runtime/src/lmeshloader.h"
#include "runtime/src/swiftshim.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// exposing pre-processor macro constants
int SEEK_SET_value () { return SEEK_SET; }
int SEEK_CUR_value () { return SEEK_CUR; }

FILE *get_stderr () { return stderr; }
]],
"-I"..liszt_dir
)
