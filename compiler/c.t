-- location to put all C includes
-- the rest of the compiler should just terralib.require this file
return terralib.includecstring [[
#include "runtime/src/lmeshloader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// exposing pre-processor macro constants
int SEEK_SET_value () { return SEEK_SET; }
int SEEK_CUR_value () { return SEEK_CUR; }

FILE *get_stderr () { return stderr; }
]]
