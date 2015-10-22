# Copyright 2015 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG?=1                   # Include debugging symbols
OUTPUT_LEVEL?=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL?=1	       # Use the shared low level

# Put the binary file name here
OUTFILE   := libmapper.so
# List all the application source files here
GEN_SRC	  := ebb_mapper.cc  # .cc files
GEN_OBJS  := $(GEN_SRC:.cc=.o)

CC_FLAGS += -fPIC
INC_FLAGS += -I$(LG_RT_DIR) -I$(LG_RT_DIR)/realm

LD_FLAGS += -L$(LEGION_BIND_DIR)
ifeq ($(shell uname), Darwin)
	LD_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
	LD_FLAGS += -lrt lpthread
else
	LD_FLAGS += -Wl,-rpath=$(LEGION_BIND_DIR)
	LD_FLAGS += -shared
	LD_FLAGS += -lpthread
endif
LD_FLAGS += -ldl -rdynamic

all: $(OUTFILE)

$(OUTFILE) : $(GEN_OBJS)
	$(CC) -o $(OUTFILE) $(GEN_OBJS) $(LD_FLAGS) $(LIBLEGION)

$(GEN_OBJS) : %.o : %.cc
	$(CC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

clean:
	-rm *.o *.so
