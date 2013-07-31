.PHONY= all single clean

UNAME = $(shell uname)
CXX = clang

SINGLE_LIB =
SINGLE_LIB_DEP_ALL =
SINGLE_LIB_DEP_B =
SINGLE_LIB_DEP_C =
SINGLE_SRC =
CXXFLAGS_LIB =
CXXFLAGS_OBJ =
CXXFLAGS_INC = -I.. -I.

ifeq ($(UNAME),Darwin)
	SINGLE_LIB += single/runtime_single.dylib
	SINGLE_LIB_DEP_ALL += single/runtime_single.cpp common/liszt_runtime.h single/liszt_runtime.h
	SINGLE_LIB_DEP_B += single/runtime_single.cpp
	CXXFLAGS_LIB += -dynamiclib
endif

ifeq ($(UNAME),Linux)
	SINGLE_LIB += single/libruntime_single.so
	SINGLE_LIB_DEP_ALL += single/runtime_single.o
	SINGLE_LIB_DEP_B += single/runtime_single.o
	SINGLE_LIB_DEP_C += single/runtime_single.cpp
	SINGLE_SRC += single/runtime_single.cpp common/liszt_runtime.h single/liszt_runtime.h
	CXXFLAGS_LIB += -shared
	CXXFLAGS_OBJ += -c -fPIC -Wno-deprecated
endif

all: $(SINGLE_LIB)

$(SINGLE_LIB): $(SINGLE_LIB_DEP_ALL)
	$(CXX) $(CXXFLAGS_LIB) $(CXXFLAGS_INC) -lstdc++ $(SINGLE_LIB_DEP_B) -o $(SINGLE_LIB)

$(SINGLE_LIB_DEP_B): $(SINGLE_SRC)
ifeq ($(UNAME),Linux)
	$(CXX) $(CXXFLAGS_OBJ) $(CXXFLAGS_INC) $(SINGLE_LIB_DEP_C) -o $(SINGLE_LIB_DEP_B)
endif

clean:
	rm -rf *~ \#* single/*.o single/*.dylib single/*.so
