.PHONY= all single clean

UNAME = $(shell uname)
CXX = clang

SINGLE_LIB =

SINGLE_DEP =
SINGLE_SRC =

COMMON_DEP =
COMMON_SRC =

CXXFLAGS_LIB =
CXXFLAGS_OBJ =
CXXFLAGS_INC = -I.. -I.

ifeq ($(UNAME),Darwin)
	SINGLE_LIB += single/runtime_single.dylib
	SINGLE_DEP += single/runtime_single.cpp
	COMMON_DEP += common/mesh.cpp
	CXXFLAGS_LIB += -dynamiclib
endif

ifeq ($(UNAME),Linux)
	SINGLE_LIB += single/libruntime_single.so
	SINGLE_DEP += single/runtime_single.o
	SINGLE_SRC += single/runtime_single.cpp
	COMMON_DEP += common/mesh.o
	COMMON_SRC += common/mesh.cpp
	CXXFLAGS_LIB += -shared
	CXXFLAGS_OBJ += -c -fPIC -Wno-deprecated
endif

all: $(SINGLE_LIB)

$(SINGLE_LIB): $(SINGLE_DEP) $(COMMON_DEP)
	$(CXX) $(CXXFLAGS_LIB) $(CXXFLAGS_INC) -lstdc++ $(SINGLE_DEP) $(COMMON_DEP) -o $(SINGLE_LIB)

$(SINGLE_DEP): $(SINGLE_SRC)
ifeq ($(UNAME),Linux)
	$(CXX) $(CXXFLAGS_OBJ) $(CXXFLAGS_INC) $(SINGLE_SRC) -o $(SINGLE_DEP)
endif

$(COMMON_DEP): $(COMMON_SRC)
ifeq ($(UNAME),Linux)
	$(CXX) $(CXXFLAGS_OBJ) $(CXXFLAGS_INC) $(COMMON_SRC) -o $(COMMON_DEP)
endif

clean:
	rm -rf *~ \#* single/*.o common/*.o single/*.dylib single/*.so
