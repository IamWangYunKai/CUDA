#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"  
int main(void){
	test();
	//cudaDeviceSynchronize();
	system("pause");
	return 0;
}