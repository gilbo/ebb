.PHONY= all single clean
.SUFFIXES:
.SECONDARY:

UNAME = $(shell uname)
CXX = clang++

COMMON_SRC = src/mesh.cpp
COMMON_OBJ = $(patsubst src/%.cpp,build/%.o,$(COMMON_SRC))

FLAGS = -I. -O3 -fPIC -Wno-deprecated

ifeq ($(UNAME),Darwin)
	LFLAGS += -dynamiclib
else
	LFLAGS += -shared
endif

LIBRARIES = libsingle_runtime.so

all: $(LIBRARIES)

lib%.so:   build/%.o $(COMMON_OBJ)
	$(CXX) $(LFLAGS) $< $(COMMON_OBJ) -o $@
    
build/%.o:	src/%.cpp 
	$(CXX) $(FLAGS) $< -c -o $@
	
clean:
	rm -rf $(LIBRARIES) build/*

OBJS = $(wildcard src/*.cpp:.cpp=.o)
DEPENDENCIES = $(patsubst %.o,build/%.d,$(OBJS))
build/%.d:	src/%.cpp
	@g++ $(FLAGS) -w -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
#if we are cleaning, then don't include dependencies (which would require the header files are built)	
ifeq ($(findstring $(MAKECMDGOALS),purge clean),)
-include $(DEPENDENCIES)
endif