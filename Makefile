
# These variables describe the location of your Terra and Legion
# installations; set them to the appropriate values please.
DEFAULT_TERRA_INSTALL_DIR:=../terra
DEFAULT_LEGION_INSTALL_DIR:=../legion

# These variables define the location of various resources
TERRA_DIR:=$(realpath ./terra)
LEGION_DIR:=$(realpath ./legion)
LUAJIT_DIR:=$(TERRA_DIR)/build/LuaJIT-2.0.3
LEGION_BIND_DIR:=$(LEGION_DIR)/bindings/terra
LIBLEGION_TERRA:=$(LEGION_BIND_DIR)/liblegion_terra.so

all: terra legion liblegion_terra
	make -C runtime

nolegion: terra
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
	LUAJIT_DIR=$(LUAJIT_DIR) TERRA_DIR=$(TERRA_DIR) make -C $(LEGION_BIND_DIR) clean
	rm terra
	rm legion
	
test: all
	terra/terra run_tests.lua
