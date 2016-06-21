#include <stdio.h>
#include <cuda.h>

__global__ void matrixAddKernel(int *a,int *b, int *c, int N)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int index = row * N + col;

    if(col < N && row < N)
    {
        c[index] = a[index]+b[index];
    } 
}

void matrixAdd(int *a, int *b, int *c, int N)
{
    int index;
    for(int col=0; col<N; col++)
    {
        for(int row=0; row<N; row++)
        {
            index = row * N + col;
            c[index] = a[index] + b[index];
        }
    } 
}

int main(int argc, char *argv[])
{
    //matrix size in each dimension
    int N = 10;

    //grid and block sizes
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    //host memory pointers
    int *a_h;
    int *b_h;
    int *c_h;
    int *d_h;

    //device memory pointers
    int *a_d;
    int *b_d;
    int *c_d;

    //number of bytes in arrays
    int size;

    //variable used for storing keyboard input
    char key;

    //CUDA events to measure time
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;

    //print out summary
    printf("Number of threads: %i (%ix%i)\n", block.x*block.y,
    block.x, block.y);
    printf("Number of blocks:  %i (%ix%i)\n", grid.x*grid.y, grid.x,
    grid.y);

    //number of bytes in each array
    size = N * N * sizeof(int);

    //allocate memory on host, this time we are using dynamic
    //allocation
    a_h = (int*) malloc(size);
    b_h = (int*) malloc(size);
    c_h = (int*) malloc(size);
    d_h = (int*) malloc(size);

    //load arrays with some numbers
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            a_h[i * N + j] = i;
            b_h[i * N + j] = i;
        } 
    }

    //GPU computation//////////////////////////////////
    //allocate device memory
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&c_d, size);

    //copy the host arrays to device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

    //start timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //launch kernel
    matrixAddKernel<<<grid, block>>>(a_d, b_d, c_d, N);

    //stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    //print out execution time
    printf("Time to calculate results on GPU: %f ms.\n", elapsedTime);

    //copy the results to host
    cudaMemcpy(c_h, c_d, size ,cudaMemcpyDeviceToHost);

    //grid and block sizes
    //CPU computation//////////////////////////////////
    //start timer
    cudaEventRecord(start, 0);

    //do the calculation on host
    matrixAdd(a_h, b_h, d_h, N);



    //stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop );

    //print out execution time
    printf("Time to calculate results on CPU: %f ms.\n", elapsedTime);

    //check if the CPU and GPU results match
    for(int i=0; i<N*N; i++)
    {
        if (c_h[i] != d_h[i]) printf("Error: CPU and GPU results do not match\n");
            break;
    }

    //clean up
    free(a_h);
    free(b_h);
    free(c_h);
    free(d_h);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0; 
}
