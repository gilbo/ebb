#include "stdio.h"

#include <cuda_runtime.h>

static const int N=100000;

/* 
	Saxpy: Z = a * X + Y, where
	- all variables are single precision,
	- a is a constant
	- X is a vector
	- Y is a vector
*/
__global__ void saxpy (float a, float *x, float *y, float *z, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		z[i] = a * x[i] + y[i];
	}
}

#define CUDA_CHECK(str) 														\
	if (err != cudaSuccess) {													\
		fprintf(stderr, str);													\
		fprintf(stderr, "\n    (error code %s)\n", cudaGetErrorString(err));	\
		exit(-1);																\
	}

int main (int argc, char **argv) {
	float *hostX = (float*) malloc(N*sizeof(float));
	float *hostY = (float*) malloc(N*sizeof(float));

	for (int i = 0; i < N; i++) {
		hostX[i] = i;
		hostY[i] = i*i;
	}

	cudaError_t err = cudaSuccess;

	// Allocate device arrays
	float *devX, *devY, *devZ;
	err = cudaMalloc((void**)&devX, N*sizeof(float));
	CUDA_CHECK("Failed to allocate device vector X!");
	err = cudaMalloc((void**)&devY, N*sizeof(float));
	CUDA_CHECK("Failed to allocate device vector Y!");
	err = cudaMalloc((void**)&devZ, N*sizeof(float));
	CUDA_CHECK("Failed to allocate device vector Z!");

	// Copy host array contents to device
	err = cudaMemcpy(devX, hostX, N*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK("Failed to copy vector X from host to device!");
	err = cudaMemcpy(devY, hostY, N*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK("Failed to copy vector Y from host to device!");

	// Launch saxpy kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	float a = 3.49230;
	saxpy<<<blocksPerGrid, threadsPerBlock>>>(a, devX, devY, devZ, N);
	err = cudaGetLastError();
	CUDA_CHECK("Failed to launch saxpy kernel!");

	// Copy result back to host
	float *hostZ = (float*) malloc(N*sizeof(float));
	err = cudaMemcpy(hostZ, devZ, N*sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK("Failed to copy vector Z from device to host!");

	// Verify correctness of result
	for (int i = 0; i < N; i++) {
		float ex = (a * hostX[i] + hostY[i]);
		float err = (hostZ[i] - ex)/ex;
		if (err < 0) err = -err;
		if (err > 1e-6) {
			fprintf(stderr, "Result failed at element %d (dev: %f, host: %f, error: %f)\n", i, hostZ[i], ex, err);
			exit(-1);
		}
	}

	// Free all allocated memory
	free(hostX);    free(hostY);    free(hostZ);
	cudaFree(devX); cudaFree(devY); cudaFree(devZ);
}