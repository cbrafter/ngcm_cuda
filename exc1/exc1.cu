#include<stdio.h>

__global__ voidmykernel(void) 
{} 

intmain(void) 
{ 
	mykernel<<<1,1>>>(); 
	printf("Hello World!\n"); 
	return 0; 
}