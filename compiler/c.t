-- location to put all C includes
-- the rest of the compiler should just terralib.require this file
return terralib.includecstring [[
#include "runtime/src/lmeshloader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

FILE *get_stderr () { return stderr; }
]]