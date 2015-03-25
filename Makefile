
# These variables describe the location of your Terra and Legion
#   installations; set them to the appropriate values please.
# If Legion isn't available, don't worry,
#   this Makefile won't try to build Legion support.
TERRA_DIR?=../terra
LEGION_DIR?=../legion

# Check whether or not the directory specified
# exists or not
FOUND_TERRA:=$(wildcard $(TERRA_DIR))
FOUND_LEGION:=$(wildcard $(LEGION_DIR))

# report an error to the user if we can't find Terra
ifndef FOUND_TERRA
define TERRA_NOT_FOUND_ERROR
Could not find Terra in the default location:
  $(TERRA_DIR)
Please edit the Makefile to specify where Terra is located
endef
$(error $(TERRA_NOT_FOUND_ERROR))
endif
# Do not report an error if we can't find Legion.  That's ok.


# Using the above info, we want to create a variable containing
# the full path string to Terra and Legion respectively
# because we may not have a Legion directory, and
# because we may not have installed symlinks, this is a little bit long
REAL_TERRA_DIR:=$(realpath ./terra)
REAL_LEGION_DIR:=$(realpath ./legion)
#ifdef here protects against case that symlinks do not exist yet
ifdef FOUND_LEGION
ifndef REAL_LEGION_DIR
REAL_LEGION_DIR:=$(realpath $(LEGION_DIR))
endif
endif
ifndef REAL_TERRA_DIR
REAL_TERRA_DIR:=$(realpath $(TERRA_DIR))
endif

# Locations of various directories needed for the Legion build
LUAJIT_DIR:=$(REAL_TERRA_DIR)/build/LuaJIT-2.0.3
LEGION_BIND_DIR:=$(REAL_LEGION_DIR)/bindings/terra
LIBLEGION_TERRA:=$(LEGION_BIND_DIR)/liblegion_terra.so
# environment variables to be set for recursive call to Legion build
SET_ENV_VAR:=LUAJIT_DIR=$(LUAJIT_DIR) TERRA_DIR=$(REAL_TERRA_DIR)


# # ----------------------------------------------------------------------- # #

# Depending on whether we're building legion, modify the
# set of build dependencies
ALL_DEP:= terra
ifdef FOUND_LEGION
ALL_DEP:=$(ALL_DEP) legion liblegion_terra
endif

all: $(ALL_DEP)
	make -C runtime


# these are targets to setup symlinks, not build things
terra:
	ln -s $(TERRA_DIR) $@

legion:
	ln -s $(LEGION_DIR) $@


# this is a target to build only those parts of legion we need
liblegion_terra: terra legion
	$(SET_ENV_VAR) make -C $(LEGION_BIND_DIR) 


# undo anything that this makefile might have done
clean:
	make -C runtime clean
ifdef REAL_LEGION_DIR # don't try to recursively call into nowhere
	$(SET_ENV_VAR) make -C $(LEGION_BIND_DIR) clean
endif
	-rm legion
	-rm terra


test: all
	@echo "\n"
	./run_tests --help
	@echo "\n** Please call the test script directly **\n"

