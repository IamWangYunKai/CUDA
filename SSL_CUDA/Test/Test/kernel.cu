#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
//void test(const double *ourplay_x, const double *ourplay_y, const double *ourplay_dir, const double *ourplay_vx, const double *ourplay_vy, const double *ourplay_vr, unsigned int size);

void test(const double *ourplay_x, const double *ourplay_y, const double *ourplay_dir, const double *ourplay_vx, const double *ourplay_vy, const double *ourplay_vr, unsigned int size) {
	double *dev_ourplay_x;
	double *dev_ourplay_y;
	double *dev_ourplay_dir;
	double *dev_ourplay_vx;
	double *dev_ourplay_vy;
	double *dev_ourplay_vr;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_ourplay_x, size * sizeof(double));
	cudaMalloc((void**)&dev_ourplay_y, size * sizeof(double));
	cudaMalloc((void**)&dev_ourplay_dir, size * sizeof(double));
	cudaMalloc((void**)&dev_ourplay_vx, size * sizeof(double));
	cudaMalloc((void**)&dev_ourplay_vy, size * sizeof(double));
	cudaMalloc((void**)&dev_ourplay_vr, size * sizeof(double));
	cudaMemcpy(dev_ourplay_x, ourplay_x, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ourplay_y, ourplay_y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ourplay_dir, ourplay_dir, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ourplay_vx, ourplay_vx, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ourplay_vy, ourplay_vy, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ourplay_vr, ourplay_vr, size * sizeof(double), cudaMemcpyHostToDevice);
	//do nothing
	cudaFree(dev_ourplay_x);
	cudaFree(dev_ourplay_y);
	cudaFree(dev_ourplay_dir);
	cudaFree(dev_ourplay_vx);
	cudaFree(dev_ourplay_vy);
	cudaFree(dev_ourplay_vr);
}

void main() {
	const unsigned int num = 12;
	double ourplay_x[num] = { 0 };
	double ourplay_y[num] = { 0 };
	double ourplay_dir[num] = { 0 };
	double ourplay_vx[num] = { 0 };
	double ourplay_vy[num] = { 0 };
	double ourplay_vr[num] = { 0 };
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	test(ourplay_x, ourplay_y, ourplay_dir, ourplay_vx, ourplay_vy, ourplay_vr, num);
	cudaEventRecord(stop, 0);
	//confirm that all things have been done before "stop event"
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	printf("Time for I/O : %.5f ms\n", elapseTime);
	system("pause");
}