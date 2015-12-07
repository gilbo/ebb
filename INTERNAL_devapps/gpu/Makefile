# Cuda path should already be set for linux
CUDA_PATH ?= /Developer/NVIDIA/CUDA-5.5

INCLUDES=-I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc
LDFLAGS=-L$(CUDA_PATH)/lib --gpu-architecture=sm_20

.PHONY: all

all: deviceInfo saxpy ptx

deviceInfo: deviceInfo.cu
	nvcc $(INCLUDES) $(LDFLAGS) deviceInfo.cu -o deviceInfo

saxpy: saxpy.cu
	nvcc $(INCLUDES) $(LDFLAGS) saxpy.cu -o saxpy

ptx: saxpy.ptx

saxpy.ptx: saxpy.cu
	nvcc $(INCLUDES) $(LDFLAGS) --ptx saxpy.cu -o saxpy.ptx


clean:
	@rm -f saxpy deviceInfo
