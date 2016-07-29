// The MIT License (MIT)
// 
// Copyright (c) 2016 Stanford University.
// All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// This include also brings in the Terra and Lua header files
#include "ebb.h"

static void doerror(lua_State * L) {
    printf("%s\n",luaL_checkstring(L,-1));
    exit(1);
}

int main(int argc, char ** argv) {
    // Create a Lua interpreter
    lua_State * L = luaL_newstate();
    // and install the standard Lua libraries
    luaL_openlibs(L);

    // In order to get good GPU performance, we need to tell Terra
    // to use mcjit.  Normally, the Ebb REPL would set this option for us,
    // but now we're taking responsibility for initializing the interpreter.
    terra_Options terra_options;
    memset(&terra_options, 0, sizeof(terra_Options));
    terra_options.usemcjit = 1;
    if(terra_initwithoptions(L, &terra_options))
        doerror(L);

    // To use Ebb, we must initialize the interpreter state with the
    // location of the Ebb compiler files and any options we wish to set
    EbbOptions ebb_options;
    ebb_options.include_dir = "../../include";
    ebb_options.use_gpu     = 0;
    if(setupebb(L, &ebb_options))
        doerror(L);

    // Of course, this is a regular Lua/Terra interpreter, so we
    // can simply submit scripts as we would any other Terra script
    if (terra_dostring(L, "print('hello world')"))
        doerror(L);

    // If we want to specifically submit an Ebb script, we simply
    // make sure the chunk begins with "import 'ebb'" like a
    // normal ebb script file.
    if (terra_dostring(L,
      "import 'ebb'\n"
      "local L = require 'ebblib'\n"
      "SAVE = {}\n"
      "SAVE.cells = L.NewRelation{ name='cells', size=4 }\n"

      "ebb SAVE.printsum( c : SAVE.cells )\n"
      "  L.print(21 + 21)\n"
      "end\n"
      "SAVE.cells:foreach(SAVE.printsum)\n"
    )) doerror(L);

    // Since we stuck some of the objects from that last script
    // into a global table SAVE, they should remain
    // accessible and re-executable from anther script.
    if (terra_dostring(L, "SAVE.cells:foreach(SAVE.printsum)"))
        doerror(L);

    // In this way, we can execute scripts for individual simulation
    // iterations and return program control back to the driving C/C++
    // program in-between interpreter invocations.

    // If we return a value from the script, then it will be placed
    // on the Lua stack.  We can then retreive the value.  For instance:
    printf("size of Lua stack: %d\n", lua_gettop(L));
    if (terra_dostring(L, "return SAVE.cells:Size()"))
        doerror(L);
    double n_cells = lua_tonumber(L, -1);
    lua_pop(L,1); // clean up the stack
    printf("number of simulation cells: %f\n", n_cells);
    printf("size of Lua stack: %d\n", lua_gettop(L));

    // As you can see from the code in this tutorial, Ebb defers its
    // C-interoperability semantics to Terra, which largely defers its
    // own C-interoperability semantics to Lua/LuaJIT.  While this may
    // seem needlessly complex at first, the Lua-C interoperability
    // mechanisms were carefully designed, and have lots of existing
    // documentation online in the Lua community.
    
    // Here are some excellent resources and documentation to start with:
    //  - Terra C API documentation:
    //        http://terralang.org/api.html#c-api
    //  - Lua C API documentation:
    //        https://www.lua.org/manual/5.2/manual.html#4
    //  - the Ebb REPL can be found in EBB_ROOT/src_interpreter.  It's a
    //      modification of the Terra REPL, which is a modification of the
    //      Lua REPL.  It has some useful examples of setting up more
    //      robust error handling.
    
    return 0;
}



