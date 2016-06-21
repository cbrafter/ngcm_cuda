#include <stdio.h>

__global__ void helloWorld(float f) 
{
	/*printf("Hello thread %d, f=%f\n", threadIdx.x, f);*/
	/* printf("Hello block %i running thread %i, f=%f\n", blockIdx.x, threadIdx.x, f);*/
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Hello block %i running thread %i, f=%f\n", blockIdx.x, idx, f);
}

int main() 
{
	helloWorld<<<1, 10>>>(1.2345f); 
	cudaDeviceReset();
	return 0;
}