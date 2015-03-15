
# These variables describe the location of your Terra and Legion
# installations; set them to the appropriate values please.
# If Legion isn't available, don't worry.
# This Makefile will not try to build Legion support.
DEFAULT_TERRA_INSTALL_DIR:=../terra
DEFAULT_LEGION_INSTALL_DIR:=../legion

FOUND_TERRA:=$(wildcard $(DEFAULT_TERRA_INSTALL_DIR))
FOUND_LEGION:=$(wildcard $(DEFAULT_LEGION_INSTALL_DIR))

# report an error to the user if we can't find Terra
ifndef FOUND_TERRA
define TERRA_NOT_FOUND_ERROR
Could not find Terra in the default location:
  $(DEFAULT_TERRA_INSTALL_DIR)
Please edit the Makefile to specify where Terra is located
endef
$(error $(TERRA_NOT_FOUND_ERROR))
endif
# Do not report an error if we can't find Legion.  That's ok.



# These variables define the location of various resources
TERRA_DIR:=$(realpath ./terra)
LEGION_DIR:=$(realpath ./legion)
#ifdef here protects against case that symlinks do not exist yet
ifdef FOUND_LEGION
ifndef LEGION_DIR
LEGION_DIR:=$(realpath $(DEFAULT_LEGION_INSTALL_DIR))
endif
endif
ifdef FOUND_TERRA
ifndef TERRA_DIR
TERRA_DIR:=$(realpath $(DEFAULT_TERRA_INSTALL_DIR))
endif
endif

LUAJIT_DIR:=$(TERRA_DIR)/build/LuaJIT-2.0.3
LEGION_BIND_DIR:=$(LEGION_DIR)/bindings/terra
LIBLEGION_TERRA:=$(LEGION_BIND_DIR)/liblegion_terra.so


# # ----------------------------------------------------------------------- # #


ALL_DEP:= terra
ifdef FOUND_LEGION
ALL_DEP:=$(ALL_DEP) legion liblegion_terra
endif

all: $(ALL_DEP)
	make -C runtime


# these are targets to setup symlinks, not build things
terra:
	ln -s $(DEFAULT_TERRA_INSTALL_DIR) $@

legion:
	ln -s $(DEFAULT_LEGION_INSTALL_DIR) $@


# this is a target to build only those parts of legion we need
liblegion_terra: terra legion
	LUAJIT_DIR=$(LUAJIT_DIR) TERRA_DIR=$(TERRA_DIR) make -C $(LEGION_BIND_DIR) 


# undo anything that this makefile might have done
clean:
	make -C runtime clean
ifdef LEGION_DIR
	LUAJIT_DIR=$(LUAJIT_DIR) TERRA_DIR=$(TERRA_DIR) make -C $(LEGION_BIND_DIR) clean
	rm legion
endif
ifdef TERRA_DIR
	rm terra
endif


test: all
	terra/terra run_tests.lua

