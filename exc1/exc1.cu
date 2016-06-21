#include <stdio.h>
__global__ void helloWorld(float f) 
{
	printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main() 
{
	helloWorld<<<1, 10>>>(1.2345f); 
	cudaDeviceReset();
	return 0;
}