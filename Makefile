
# The MIT License (MIT)
# 
# Copyright (c) 2015 Stanford University.
# All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# # ----------------------------------------------------------------------- # #
#     User Configurable Makefile variables

TERRA_DIR?=../terra/release
# The following variables are only necessary for Legion development
LEGION_DIR?=../legion
TERRA_ROOT_DIR?=../terra


# # ----------------------------------------------------------------------- # #
#     Detect Platform
PLATFORM:=UNKNOWN
ifeq ($(OS),Windows_NT)
    PLATFORM:=WINDOWS
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        PLATFORM:=LINUX
    endif
    ifeq ($(UNAME_S),Darwin)
        PLATFORM:=OSX
    endif
endif
# Reject Currently Unsupported Platforms
ifneq ($(PLATFORM),LINUX)
  ifneq ($(PLATFORM), OSX)
    $(error The Build Process has not been tested for $(PLATFORM))
  endif
endif

# test for Mac OS X command line tools
# and issue debugging information for common problems
ifeq ($(PLATFORM),OSX)
  CMD_LINE_TOOLS_INSTALLED:=$(shell xcode-select -p >/dev/null; echo $$?)
  ifneq ($(CMD_LINE_TOOLS_INSTALLED),0)
  	$(error \
  		The Command Line Tools were not found, or their installation was \
  		incomplete; Executing 'xcode-select --install' may fix the problem \
  	 )
  endif
endif

# # ----------------------------------------------------------------------- # #
#     Terra-Setup

# Detect Whether or not Terra is installed and in what way
TERRA_DIR_EXISTS:=$(wildcard $(TERRA_DIR))
TERRA_SYMLINK_EXISTS:=$(wildcard terra)
# Then make sure we can see Terra as a subdir in the expected way
ifndef TERRA_SYMLINK_EXISTS
ifndef TERRA_DIR_EXISTS
  DOWNLOAD_TERRA=1
else
  MAKE_TERRA_SYMLINK=1
endif
endif

OSX_TERRA_NAME=terra-OSX-x86_64-84bbb0b
OSX_TERRA_URL=https://github.com/zdevito/terra/releases/download/release-2015-08-03/$(OSX_TERRA_NAME).zip
LINUX_TERRA_NAME=terra-Linux-x86_64-84bbb0b
LINUX_TERRA_URL=https://github.com/zdevito/terra/releases/download/release-2015-08-03/$(LINUX_TERRA_NAME).zip

# # ----------------------------------------------------------------------- # #
#     Legion-Specific Setup

# Unlike Terra, we won't download Legion automatically.
# However, if we can't find Legion, that's fine
LEGION_DIR_EXISTS:=$(wildcard $(LEGION_DIR))
LEGION_SYMLINK_EXISTS:=$(wildcard legion)
ifdef LEGION_SYMLINK_EXISTS
  LEGION_INSTALLED=1
else
  ifdef LEGION_DIR_EXISTS
    MAKE_LEGION_SYMLINK=1
    LEGION_INSTALLED=1
  endif
endif

ifdef LEGION_INSTALLED
  REAL_TERRA_DIR:=$(realpath $(TERRA_ROOT_DIR))
  REAL_LEGION_DIR:=$(realpath $(LEGION_DIR))

  # Determine whether or not CUDA is available and visible to Legion
  USE_CUDA=0
  ifdef CUDA
  USE_CUDA=1
  endif
  ifdef CUDATOOLKIT_HOME
  USE_CUDA=1
  endif

  # Locations of various directories needed for the Legion build
  LUAJIT_DIR:=$(REAL_TERRA_DIR)/build/LuaJIT-2.0.3
  LEGION_BIND_DIR:=$(REAL_LEGION_DIR)/bindings/terra
  LEGION_RUNTIME_DIR:=$(REAL_LEGION_DIR)/runtime
  LIBLEGION_TERRA:=$(LEGION_BIND_DIR)/liblegion_terra.so
  LIBLEGION_TERRA_RELEASE:=$(LEGION_BIND_DIR)/liblegion_terra_release.so
  LIBLEGION_TERRA_DEBUG:=$(LEGION_BIND_DIR)/liblegion_terra_debug.so
  # environment variables to be set for recursive call to Legion build
  SET_ENV_VAR:=LUAJIT_DIR=$(LUAJIT_DIR) TERRA_DIR=$(REAL_TERRA_DIR) \
    LG_RT_DIR=$(REAL_LEGION_DIR)/runtime SHARED_LOWLEVEL=0 USE_GASNET=0 USE_CUDA=$(USE_CUDA)
endif # LEGION_INSTALLED

# # ----------------------------------------------------------------------- # #
#     Interpreter

DYNLIBTERRA=terra/libterra.so
LIBTERRA=terra/lib/libterra.a

EXECUTABLE=ebb
EXEC_OBJS = main.o linenoise.o
# Use default CXX

INTERP_CFLAGS = -Wall -g -fPIC
INTERP_CFLAGS += -I build -I src_interpreter -I terra/include/terra
INTERP_CFLAGS += -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -O0 -fno-rtti -fno-common -Woverloaded-virtual -Wcast-qual -fvisibility-inlines-hidden

INTERP_LFLAGS = -g
ifeq ($(PLATFORM),OSX)
  INTERP_LFLAGS += -pagezero_size 10000 -image_base 100000000
endif
ifeq ($(PLATFORM),LINUX)
  INTERP_LFLAGS += -Wl,-export-dynamic -Wl,--whole-archive $(LIBTERRA) -Wl,--no-whole-archive
  INTERP_LFLAGS += -ldl -pthread
else
  INTERP_LFLAGS += -Wl,-force_load,$(LIBTERRA)
endif
# Can we always add these for safety?
INTERP_LFLAGS += -lcurses -lz

# # ----------------------------------------------------------------------- # #
#     Legion Mapper

ifdef LEGION_INSTALLED

  MAPPER_OBJS := mappers/ebb_mapper.o
  LIBMAPPER:=mappers/libmapper.so
  LIBMAPPER_RELEASE:=mappers/libmapper_release.so
  LIBMAPPER_DEBUG:=mappers/libmapper_debug.so

  MAPPER_CFLAGS:=-fPIC
  MAPPER_CFLAGS+=-I$(LEGION_RUNTIME_DIR) -I$(LEGION_RUNTIME_DIR)/legion -I$(LEGION_RUNTIME_DIR)/mappers -I$(LEGION_RUNTIME_DIR)/realm

  MAPPER_LFLAGS:=-L$(LEGION_BIND_DIR)
  ifeq ($(PLATFORM),OSX)
    MAPPER_LFLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
  endif
  ifeq ($(PLATFORM),LINUX)
    MAPPER_LFLAGS += -Wl,-rpath=$(LEGION_BIND_DIR)
    MAPPER_LFLAGS += -shared
  endif
  MAPPER_LFLAGS += -ldl -lpthread -rdynamic

endif # LEGION_INSTALLED

# # ----------------------------------------------------------------------- # #
#     Legion Utils

ifdef LEGION_INSTALLED

  LEGION_UTILS_SRC := $(wildcard legion_utils/*.cc)
  LEGION_UTILS_H := $(wildcard legion_utils/*.h)
  LEGION_UTILS_OBJS := $(subst .cc,.o,$(LEGION_UTILS_SRC))
  LIBLEGION_UTILS:=legion_utils/liblegion_utils.so
  LIBLEGION_UTILS_RELEASE:=legion_utils/liblegion_utils_release.so
  LIBLEGION_UTILS_DEBUG:=legion_utils/liblegion_utils_debug.so

  LEGION_UTILS_CFLAGS:=-fPIC
  LEGION_UTILS_CFLAGS+=-I$(LEGION_RUNTIME_DIR) -I$(LEGION_RUNTIME_DIR)/legion -I$(LEGION_RUNTIME_DIR)/mappers -I$(LEGION_RUNTIME_DIR)/realm

  LEGION_UTILS_LFLAGS:=-L$(LEGION_BIND_DIR)
  ifeq ($(PLATFORM),OSX)
    LEGION_UTILS_LFLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
  endif
  ifeq ($(PLATFORM),LINUX)
    LEGION_UTILS_LFLAGS += -Wl,-rpath=$(LEGION_BIND_DIR)
    LEGION_UTILS_LFLAGS += -shared
  endif
  LEGION_UTILS_LFLAGS += -ldl -lpthread -rdynamic

endif # LEGION_INSTALLED


# # ----------------------------------------------------------------------- # #
# # ----------------------------------------------------------------------- # #
#     Main Rules
# # ----------------------------------------------------------------------- # #

# Depending on whether we're building legion, modify the
# set of build dependencies
ALL_DEP:= terra $(EXECUTABLE) $(EXECUTABLE_CP)
ifdef LEGION_INSTALLED
ALL_DEP += legion $(LIBLEGION_TERRA_DEBUG) $(LIBLEGION_TERRA_RELEASE)
ALL_DEP += $(LIBMAPPER_DEBUG) $(LIBMAPPER_RELEASE)
ALL_DEP += $(LIBLEGION_UTILS_DEBUG) $(LIBLEGION_UTILS_RELEASE)
endif

.PHONY: all clean test lmesh

all: $(ALL_DEP)

# This is a deprecated legacy build
lmesh:
	$(MAKE) -C deprecated/deprecated_runtime

# auto-download rule, or make symlink to local copy rule
terra:
ifdef DOWNLOAD_TERRA
ifeq ($(PLATFORM),LINUX)
	wget -O $(LINUX_TERRA_NAME).zip $(LINUX_TERRA_URL)
	unzip $(LINUX_TERRA_NAME).zip
	mv $(LINUX_TERRA_NAME) terra
	rm $(LINUX_TERRA_NAME).zip
endif
ifeq ($(PLATFORM),OSX)
	wget -O $(OSX_TERRA_NAME).zip $(OSX_TERRA_URL)
	unzip $(OSX_TERRA_NAME).zip
	mv $(OSX_TERRA_NAME) terra
	rm $(OSX_TERRA_NAME).zip
endif
else
ifdef MAKE_TERRA_SYMLINK
	ln -s $(TERRA_DIR) terra
endif
endif

test: all
	@echo "\n"
	./run_tests --help
	@echo "\n** Please call the test script directly **\n"

# # ----------------------------------------------------------------------- # #
#     VDB Rules

# Rules to download and build the visualization tool
vdb-win:
	wget -O vdb-win.zip https://github.com/zdevito/vdb/blob/binaries/vdb-win.zip?raw=true
	unzip vdb-win.zip
	rm vdb-win.zip

vdb-src:
	wget -O vdb-src.zip https://github.com/zdevito/vdb/archive/ec461393b22c0e2f7fdce4c2d4f410f8cc671c80.zip
	unzip vdb-src.zip
	mv vdb-ec461393b22c0e2f7fdce4c2d4f410f8cc671c80 vdb-src
	rm vdb-src.zip
	make -C vdb-src
	ln -s vdb-src/vdb vdb

ifeq ($(PLATFORM),OSX)
vdb: vdb-src
endif
ifeq ($(PLATFORM),LINUX)
vdb: vdb-src
endif
ifeq ($(PLATFORM),WINDOWS)
vdb: vdb-win
endif

# # ----------------------------------------------------------------------- # #
#     Interpreter Rules

build/%.o:  src_interpreter/%.cpp terra
	mkdir -p build
	$(CXX) $(INTERP_CFLAGS) $< -c -o $@

bin/$(EXECUTABLE): $(addprefix build/, $(EXEC_OBJS)) terra
	mkdir -p bin
	$(CXX) $(addprefix build/, $(EXEC_OBJS)) -o $@ $(INTERP_LFLAGS)

$(EXECUTABLE): bin/$(EXECUTABLE)
	ln -sf bin/$(EXECUTABLE) $(EXECUTABLE)

# # ----------------------------------------------------------------------- # #
#     Legion Rules

ifdef LEGION_INSTALLED
#legion_refresh:
#	$(SET_ENV_VAR) $(MAKE) -C $(LEGION_BIND_DIR)
#	mv $(LIBLEGION_TERRA) $(LIBLEGION_TERRA_DEBUG)

legion:
	ln -s $(LEGION_DIR) legion

# this is a target to build only those parts of legion we need

$(LIBLEGION_TERRA_RELEASE): terra legion $(LIBLEGION_TERRA_DEBUG)
	$(SET_ENV_VAR) $(MAKE) -C $(LEGION_BIND_DIR) clean
	$(SET_ENV_VAR) DEBUG=0 $(MAKE) -C $(LEGION_BIND_DIR)
	mv $(LIBLEGION_TERRA) $(LIBLEGION_TERRA_RELEASE)

$(LIBLEGION_TERRA_DEBUG): terra legion
	$(SET_ENV_VAR) $(MAKE) -C $(LEGION_BIND_DIR) clean
	$(SET_ENV_VAR) CC_FLAGS=-DLEGION_SPY $(MAKE) -C $(LEGION_BIND_DIR)
	mv $(LIBLEGION_TERRA) $(LIBLEGION_TERRA_DEBUG)

endif

# # ----------------------------------------------------------------------- # #
#     Mapper (Legion) Rules

ifdef LEGION_INSTALLED

mappers/%.o: mappers/%.cc mappers/%.h
	echo $(MAPPER_CFLAGS)
	$(CXX) $(MAPPER_CFLAGS) $< -c -o $@

$(LIBMAPPER_RELEASE): $(LIBLEGION_TERRA_RELEASE) $(LIBMAPPER_DEBUG) $(MAPPER_OBJS)
	#$(SET_ENV_VAR) LIBLEGION=-llegion_terra_release DEBUG=0 $(MAKE) -C mappers
	#mv $(LIBMAPPER) $(LIBMAPPER_RELEASE)
	DEBUG=0 $(CXX) $(MAPPER_OBJS) -o $@ $(MAPPER_LFLAGS)

$(LIBMAPPER_DEBUG): $(LIBLEGION_TERRA_DEBUG) $(MAPPER_OBJS)
	#$(SET_ENV_VAR) LIBLEGION=-llegion_terra_debug $(MAKE) -C mappers
	#mv $(LIBMAPPER) $(LIBMAPPER_DEBUG)
	$(CXX) $(MAPPER_OBJS) -o $@ $(MAPPER_LFLAGS)

endif

# # ----------------------------------------------------------------------- # #
#     Legion utils (Legion) Rules

ifdef LEGION_INSTALLED

$(LIBLEGION_UTILS_RELEASE): $(LEGION_UTILS_OBJS) $(LIBLEGION_TERRA_RELEASE) $(LIBLEGION_UTILS_DEBUG)
	DEBUG=0 $(CXX) $(LEGION_UTILS_OBJS) -o $@ $(LEGION_UTILS_LFLAGS)

$(LIBLEGION_UTILS_DEBUG): $(LEGION_UTILS_OBJS) $(LIBLEGION_TERRA_DEBUG)
	$(CXX) $(LEGION_UTILS_OBJS) -o $@ $(LEGION_UTILS_LFLAGS)

$(LEGION_UTILS_OBJS): %.o : %.cc
	echo $(LEGION_UTILS_CFLAGS)
	$(CXX) $(LEGION_UTILS_CFLAGS) -c $< -o $@


endif

# # ----------------------------------------------------------------------- # #
#     Cleanup

mapperclean:
ifdef LEGION_SYMLINK_EXISTS # don't try to recursively call into nowhere
	-rm -f mappers/*.o
	-rm -f $(LIBMAPPER_DEBUG)
	-rm -f $(LIBMAPPER_RELEASE)
endif

legionutilsclean:
ifdef LEGION_SYMLINK_EXISTS # don't try to recursively call into nowhere
	-rm -f legion_utils/*.o
	-rm -f $(LIBLEGION_UTILS_DEBUG)
	-rm -f $(LIBLEGION_UTILS_RELEASE)
endif

clean: mapperclean legionutilsclean
	$(MAKE) -C deprecated/deprecated_runtime clean
	-rm -r vdb*
	-rm -r bin
	-rm -r build
ifdef LEGION_SYMLINK_EXISTS # don't try to recursively call into nowhere
	$(SET_ENV_VAR) $(MAKE) -C $(LEGION_BIND_DIR) clean
	-rm $(LIBLEGION_TERRA_RELEASE)
	-rm $(LIBLEGION_TERRA_DEBUG)
	-rm legion
endif
	-rm terra



