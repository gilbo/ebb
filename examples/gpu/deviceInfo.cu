#include "stdio.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


int main (int argc, char **argv) {
	int devID;
	cudaDeviceProp devProp;

	devID = findCudaDevice(argc, (const char**) argv);
	checkCudaErrors(cudaGetDeviceProperties(&devProp, devID));

	printf("warpSize : %d\n",                    devProp.warpSize);
	printf("maxThreadsPerBlock : %d\n",          devProp.maxThreadsPerBlock);
	printf("maxThreadsPerMultiprocessor : %d\n", devProp.maxThreadsPerMultiProcessor);
	printf("sharedMemPerBlock : %lu\n",          devProp.sharedMemPerBlock);
	printf("regsPerBlock : %d\n",                devProp.regsPerBlock);
	printf("multiProcessorCount : %d\n",         devProp.multiProcessorCount);
}