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

#ifndef __EBB_H
#define __EBB_H

#include "terra/terra.h"

#ifdef __APPLE__
#include <sys/syslimits.h>
#define _EBB_H_MAX_PATH_LEN PATH_MAX
#else // assuming Linux here
#include <linux/limits.h>
#define _EBB_H_MAX_PATH_LEN PATH_MAX
#endif

#include <unistd.h>

struct EbbOptions {
  const char *  include_dir;  // path to the ebb include directory
  int           use_gpu;      // 0 if false, 1 if true
};

static inline
int setupebb(lua_State * L, EbbOptions * E) {
    // for composing strings
    char buffer[_EBB_H_MAX_PATH_LEN]; // plenty of room
    size_t bufsize = _EBB_H_MAX_PATH_LEN;

    // check that the Ebb include directory exists at the provided
    // path by looking for this file
    snprintf(buffer, bufsize, "%s/ebb.h", E->include_dir);
    if(access(buffer, R_OK) != 0) {
      fprintf(stderr, "Could not find Ebb include directory at:\n  %s\n",
                      E->include_dir);
      return 1;
    }
    // Then go ahead and modify the terra path so it can find the
    // compiler files
    snprintf(buffer, bufsize,
      "package.terrapath = package.terrapath..';%s/?.t;'",
      E->include_dir);
    if (terra_dostring(L, buffer)) {
        return 1;
    }

    // signal to the compiler that we want to use the GPU
    if (E->use_gpu) {
        lua_pushboolean(L, true);
        lua_setglobal(L, "EBB_USE_GPU_SIGNAL");
    }
    return 0;
}

#endif // __EBB_H
