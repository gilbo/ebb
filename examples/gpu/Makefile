# Cuda path should already be set for linux
CUDA_PATH ?= /Developer/NVIDIA/CUDA-5.5

INCLUDES=-I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc
LDFLAGS=-L$(CUDA_PATH)/lib

deviceInfo: deviceInfo.cu
	nvcc $(INCLUDES) $(LDFLAGS) deviceInfo.cu -o deviceInfo
