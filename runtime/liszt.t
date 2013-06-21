-- Link in C++ implementation of single core runtime

local runtime = terralib.includec("runtime/include/liszt_runtime.h")
terralib.linklibrary("runtime/single/runtime_single.dylib")



