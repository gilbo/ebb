.PHONY= all single clean
.SUFFIXES:
.SECONDARY:

UNAME = $(shell uname)
CXX = clang++
OPT = opt

COMMON_SRC = src/mesh.cpp src/lmeshloader.cpp
COMMON_OBJ = $(patsubst src/%.cpp,build/%.o,$(COMMON_SRC))

FLAGS = -I. -g -fPIC -Wno-deprecated

ifeq ($(UNAME),Darwin)
	LFLAGS += -dynamiclib
else
	LFLAGS += -shared
endif

CUDA_HOME ?= /usr/local/cuda

LIBRARIES = libsingle_runtime.so

# Make libdevice math symbols available if the GPU runtime is supported by terra
-include ../terra/Makefile.inc
ifeq ($(ENABLE_CUDA),1)
LIBRARIES += libdevice.bc
endif

all: $(LIBRARIES)

lib%.so:   build/%.o $(COMMON_OBJ)
	$(CXX) $(LFLAGS) $< $(COMMON_OBJ) -o $@

build/%.o:	src/%.cpp 
	$(CXX) $(FLAGS) $< -c -o $@
	
libdevice.bc: $(CUDA_HOME)/nvvm/libdevice/libdevice.compute_20.10.bc
	$(OPT) -nvvm-reflect $< -o $@

clean:
	rm -rf $(LIBRARIES) build/*

SRCS = $(wildcard src/*.cpp)
DEPENDENCIES = $(patsubst src/%.cpp,build/%.d,$(SRCS))
build/%.d:	src/%.cpp
	@g++ $(FLAGS) -w -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
#if we are cleaning, then don't include dependencies (which would require the header files are built)	
ifeq ($(findstring $(MAKECMDGOALS),purge clean),)
-include $(DEPENDENCIES)
endif