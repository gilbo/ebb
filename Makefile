
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
  LIBLEGION_TERRA:=$(LEGION_BIND_DIR)/liblegion_terra.so
  LIBLEGION_TERRA_RELEASE:=$(LEGION_BIND_DIR)/liblegion_terra_release.so
  LIBLEGION_TERRA_DEBUG:=$(LEGION_BIND_DIR)/liblegion_terra_debug.so
  # environment variables to be set for recursive call to Legion build
  SET_ENV_VAR:=LUAJIT_DIR=$(LUAJIT_DIR) TERRA_DIR=$(REAL_TERRA_DIR) \
    SHARED_LOWLEVEL=0 USE_GASNET=0 USE_CUDA=$(USE_CUDA)
endif # LEGION_INSTALLED

# # ----------------------------------------------------------------------- # #
#     Interpreter

DYNLIBTERRA=terra/libterra.so
LIBTERRA=terra/lib/libterra.a

EXECUTABLE=ebb
EXECUTABLE_CP=liszt
EXEC_OBJS = main.o linenoise.o
# Use default CXX

FLAGS = -Wall -g -fPIC
FLAGS += -I build -I src_interpreter -I terra/include/terra
FLAGS += -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -O0 -fno-rtti -fno-common -Woverloaded-virtual -Wcast-qual -fvisibility-inlines-hidden

LFLAGS = -g
ifeq ($(PLATFORM),OSX)
  LFLAGS += -pagezero_size 10000 -image_base 100000000 
endif
ifeq ($(PLATFORM),LINUX)
  LFLAGS += -Wl,-export-dynamic -Wl,--whole-archive $(LIBTERRA) -Wl,--no-whole-archive
  LFLAGS += -ldl -pthread
else
  LFLAGS += -Wl,-force_load,$(LIBTERRA)
endif
# Can we always add these for safety?
LFLAGS += -lcurses -lz
#ifeq ($(shell nm $(DYNLIBTERRA) | grep setupterm 2>&1 >/dev/null; echo $$?), 0)
#  LFLAGS += -lcurses 
#endif
#ifeq ($(shell nm $(DYNLIBTERRA) | grep compress2 2>&1 >/dev/null; echo $$?), 0)
#  LFLAGS += -lz
#endif


# # ----------------------------------------------------------------------- # #
# # ----------------------------------------------------------------------- # #
#     Main Rules
# # ----------------------------------------------------------------------- # #

# Depending on whether we're building legion, modify the
# set of build dependencies
ALL_DEP:= terra $(EXECUTABLE) $(EXECUTABLE_CP)
ifdef LEGION_INSTALLED
ALL_DEP:=$(ALL_DEP) legion $(LIBLEGION_TERRA_RELEASE) $(LIBLEGION_TERRA_DEBUG)
endif

.PHONY: all clean test lmesh

all: $(ALL_DEP)

# This is a deprecated legacy build
lmesh:
	make -C deprecated_runtime

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
#     Interpreter Rules

build/%.o:  src_interpreter/%.cpp terra
	mkdir -p build
	$(CXX) $(FLAGS) $(CPPFLAGS) $< -c -o $@

bin/$(EXECUTABLE): $(addprefix build/, $(EXEC_OBJS)) terra
	mkdir -p bin
	$(CXX) $(addprefix build/, $(EXEC_OBJS)) -o $@ $(LFLAGS)

$(EXECUTABLE): bin/$(EXECUTABLE)
	ln -sf bin/$(EXECUTABLE) $(EXECUTABLE)

bin/$(EXECUTABLE_CP): bin/$(EXECUTABLE)
	ln -sf bin/$(EXECUTABLE) bin/$(EXECUTABLE_CP)

$(EXECUTABLE_CP): bin/$(EXECUTABLE)
	ln -sf bin/$(EXECUTABLE) $(EXECUTABLE_CP)

# # ----------------------------------------------------------------------- # #
#     Legion Rules

ifdef LEGION_INSTALLED
#legion_refresh:
#	$(SET_ENV_VAR) make -C $(LEGION_BIND_DIR)
#	mv $(LIBLEGION_TERRA) $(LIBLEGION_TERRA_DEBUG)

legion:
	ln -s $(LEGION_DIR) legion

# this is a target to build only those parts of legion we need
$(LIBLEGION_TERRA_RELEASE): terra legion $(LIBLEGION_TERRA_DEBUG)
	$(SET_ENV_VAR) make -C $(LEGION_BIND_DIR) clean
	$(SET_ENV_VAR) DEBUG=0 make -C $(LEGION_BIND_DIR)
	mv $(LIBLEGION_TERRA) $(LIBLEGION_TERRA_RELEASE)

$(LIBLEGION_TERRA_DEBUG): terra legion
	$(SET_ENV_VAR) make -C $(LEGION_BIND_DIR) clean
	$(SET_ENV_VAR) CC_FLAGS=-DLEGION_SPY make -C $(LEGION_BIND_DIR)
	mv $(LIBLEGION_TERRA) $(LIBLEGION_TERRA_DEBUG)
endif

# # ----------------------------------------------------------------------- # #
#     Cleanup

clean:
	make -C deprecated_runtime clean
	-rm -r bin
	-rm -r build
ifdef LEGION_SYMLINK_EXISTS # don't try to recursively call into nowhere
	$(SET_ENV_VAR) make -C $(LEGION_BIND_DIR) clean
	-rm $(LIBLEGION_TERRA_RELEASE)
	-rm $(LIBLEGION_TERRA_DEBUG)
	-rm legion
endif
	-rm terra



