
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

# If any of the below defaults are not right for your configuration,
# Create a Makefile.inc file and set the variables in it
-include Makefile.inc

# location of terra directory in binary release, or the /release subdir in
# a custom build of Terra
TERRA_DIR?=../terra
# The following variable is needed for experimental development
E2_DIR?=


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
TERRA_DIR_EXISTS:=$(wildcard $(TERRA_DIR)/bin)
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
#     Interpreter

DYNLIBTERRA=terra/libterra.so
LIBTERRA=terra/lib/libterra.a

EXECUTABLE=ebb
EXEC_OBJS = main.o linenoise.o
# Use default CXX

INTERP_CFLAGS = -Wall -g -fPIC
INTERP_CFLAGS += -I build -I src_interpreter -I terra/include/terra
INTERP_CFLAGS += -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -O0 -fno-rtti -fno-common -Woverloaded-virtual -Wcast-qual -fvisibility-inlines-hidden
ifdef E2_DIR
  INTERP_CFLAGS += -DE2_DIR="$(E2_DIR)"
endif

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
# # ----------------------------------------------------------------------- # #
#     Main Rules
# # ----------------------------------------------------------------------- # #

# Depending on whether we're building legion, modify the
# set of build dependencies
ALL_DEP:= terra $(EXECUTABLE) $(EXECUTABLE_CP)

.PHONY: all clean test lmesh

all: $(ALL_DEP)

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
#     Cleanup


clean:
	-rm -r vdb*
	-rm -r bin
	-rm -r build
	-rm terra



