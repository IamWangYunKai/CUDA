#include <stdio.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"  
#include "CU_geometry.h"

__global__ void kernel()
{
	printf("hello world!\n");
	CGeoPoint point = CGeoPoint(450, 0);
	printf("%.2f, %.2f\n", point.x(), point.y());
}

void test() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel << <1, 1 >> >();

	cudaEventRecord(stop, 0);
	//confirm that all things have been done before "stop event"
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	printf("Time for I/O : %.5f ms\n", elapseTime);
}