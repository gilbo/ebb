---
layout: tutorial
title: "18: C Embedding"
excerpt: "Ebb is built on Lua and Terra, which makes using the Ebb runtime from a controlling C program as straightforward as library linking."
---





In this tutorial, we'll show an example of how to use Ebb from within a C/C++ program.  This process is only marginally different than using Lua (& Terra) from a C program.

That C-code can be entirely found in `18_c_embedding.c`.  It needs to be compiled into an executable.  We've included some sample compilation commands in a makefile in this directory.  Note that extra compiler flags are necessary to get Terra to work correctly.


```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// This include also brings in the Terra and Lua header files
#include "ebb.h"
```

Inside the `include/` directory of the Ebb root, there is a `ebb.h` file.  Including this file will recursively include `terra.h` and `lua.h` giving us access to all the necessary APIs.


```
static void doerror(lua_State * L) {
    printf("%s\n",luaL_checkstring(L,-1));
    exit(1);
}
```

We defined a simple error handling helper function for brevity.


```
int main(int argc, char ** argv) {
    lua_State * L = luaL_newstate();
    luaL_openlibs(L);
```

We start the program the same way any program embedding a Lua interpreter would--by creating a new Lua state and installing the standard Lua libraries on its global table.  More info about setting up a Lua interpreter can be found in the [Lua Manual](https://www.lua.org/manual/5.2/manual.html#4).


```
    terra_Options terra_options;
    memset(&terra_options, 0, sizeof(terra_Options));
    terra_options.usemcjit = 1;
    if(terra_initwithoptions(L, &terra_options))
        doerror(L);
```

Now that we have a basic interpreter state, we initialize it using the Terra initialization function.  At this point, we can pass Terra specific options to the setup function.  Here we turn on the mcjit feature, which rely on to improve the PTX/CUDA code generation.  More information about the Terra initialization can be found in the [Terra Manual](http://terralang.org/api.html#c-api)


```
    EbbOptions ebb_options;
    ebb_options.include_dir = "../../include";
    ebb_options.use_gpu     = 0;
    if(setupebb(L, &ebb_options))
        doerror(L);
```

Lastly, we setup the interpreter to support Ebb programs.  This *must* be done after Terra has already been initialized.  We provide the location of the Ebb source files and any options we want.  In this case, we indicate that we do not want to enable CUDA GPU support.  This is the only Ebb specific function provided for the embedding API.


```
    if (terra_dostring(L, "print('hello world')"))
        doerror(L);
```

Of course, this is a regular Lua/Terra interpreter, so we can simply submit scripts as we would any other Terra/Lua script.


```
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
```

If we want to specifically submit an Ebb script, we simply make sure the submitted script begins with `import 'ebb'`.  We must make sure to use the Terra execution functions, since Ebb relies on Terra support to execute.


```
    if (terra_dostring(L, "SAVE.cells:foreach(SAVE.printsum)"))
        doerror(L);
```

Since we stuck some of the objects from that last script into a global table `SAVE`, they should remain accessible and re-executable from anther script.

In this way, we can execute scripts for individual simulation iterations and return program control back to the driving C/C++ program in-between interpreter invocations.

```
    printf("size of Lua stack: %d\n", lua_gettop(L));
    if (terra_dostring(L, "return SAVE.cells:Size()"))
        doerror(L);
    double n_cells = lua_tonumber(L, -1);
    lua_pop(L,1); // clean up the stack
    printf("number of simulation cells: %f\n", n_cells);
    printf("size of Lua stack: %d\n", lua_gettop(L));

    return 0;
}
```

If we return a value from the script, then it will be placed on the Lua stack.  We can then retreive the value.  All of the code in this snippet is simply using the standard Lua API and semantics.

As you can see from the code in this tutorial, Ebb defers its C-interoperability semantics to Terra, which largely defers its own C-interoperability semantics to Lua/LuaJIT.  While this may seem needlessly complex at first, the Lua-C interoperability mechanisms were carefully designed, and have lots of existing documentation online in the Lua community.

Finally, a number of simple ideas for error handling can be found in the Ebb REPL source at `src_interpreter/main.cpp`.


