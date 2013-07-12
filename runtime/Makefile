.PHONY= all single clean

all: single

single: single/runtime_single.dylib

single/runtime_single.dylib: single/runtime_single.cpp common/liszt_runtime.h single/liszt_runtime.h
	clang -dynamiclib -I. -lstdc++ single/runtime_single.cpp -o single/runtime_single.dylib

clean:
	rm -rf *~ \#* single/*.o single/*.dylib

