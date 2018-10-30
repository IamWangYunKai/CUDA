#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

const unsigned int NUM = 12;
const unsigned int TARGRT_NUM = 30;
const unsigned int GENERATE_NUM = 1024;

typedef struct {
	double x;
	double y;
}Point;

__constant__ const Point target[TARGRT_NUM] = {
	{ 600, -58 },{ 600, -54 },{ 600, -50 },{ 600, -46 },{ 600, -42 },{ 600, -38 },{ 600, -34 },{ 600, -30 },{ 600, -26 },{ 600, -22 },
	{ 600, -18 },{ 600, -14 },{ 600, -10 },{ 600, -6 },{ 600, -2 },{ 600, 2 },{ 600, 6 },{ 600, 10 },{ 600, 14 },{ 600, 18 },
	{ 600, 22 },{ 600, 26 },{ 600, 30 },{ 600, 34 },{ 600, 38 },{ 600, 42 },{ 600, 46 },{ 600, 50 },{ 600, 54 },{ 600, 58 }
};

typedef struct {
	double x;
	double y;
	double dir;
	double vx;
	double vy;
	double vr;
}OurPlayer;

typedef struct {
	double x;
	double y;
	double dir;
}TheirPlayer;

__constant__ OurPlayer dev_ourplayer[NUM];
__constant__ TheirPlayer dev_theirplayer[NUM];
__constant__ Point dev_ball;
const int CONST_MEM_SIZE = NUM * sizeof(OurPlayer) + NUM * sizeof(TheirPlayer) + (TARGRT_NUM + 1) * sizeof(Point);

void getInf(const OurPlayer *ourplayer, const TheirPlayer *theirplayer, const Point ball) {
	cudaMemcpyToSymbol(dev_ourplayer, ourplayer, NUM * sizeof(OurPlayer));
	cudaMemcpyToSymbol(dev_theirplayer, theirplayer, NUM * sizeof(TheirPlayer));
	cudaMemcpyToSymbol(&dev_ball, &ball, sizeof(Point));
}

//分别传入当前点、射门点、阻挡点，若能射门返回true，否则返回false
bool calcShootRange(Point source, Point target, Point object) {
	double A = target.y - source.y;
	double B = source.x - target.x;
	double C = target.x * source.y - source.x * target.y;
	double D = sqrt(A*A + B*B);
	if (D < 0.01) return false;
	double dist = abs(A*object.x + B*object.y + C) / D;
	if (dist > 9.0) return true;
	return false;
}

__device__ void calcShootRangeList(Point source, TheirPlayer *object) {
	bool dev_shootList[TARGRT_NUM];
	for (int i = 0; i < TARGRT_NUM; i++) {
		dev_shootList[i] = true;//初始化
		double A = target[i].y - source.y;
		double B = source.x - target[i].x;
		double C = target[i].x * source.y - source.x * target[i].y;
		double D = sqrt(A*A + B*B);
		//D可能等于0?
		if (D < 1) {
			dev_shootList[i] = false;
			continue;
		}
		for (int j = 0; j < NUM; j++) {
			double dist = abs(A*object[j].x + B*object[j].y + C) / D;
			if (dist < 9.0) {//有阻挡
				dev_shootList[i] = false;
				break;
			}
			else continue;
		}
	}
}

//生成1024个射门点
void generatePoint(Point *shootPoint, unsigned int size) {
	for (int i = 0; i < size; i++) {
		shootPoint[i].x = rand() % 600;//生成0~600的整数
		shootPoint[i].y = (rand() % 900) -450;//生成-450~450的整数
		//printf("Point %d: (%.1f, %.1f)\n", i, shootPoint[i].x, shootPoint[i].y);
	}
}

__global__ void kernel(Point *dev_shootPoint){
	int i = threadIdx.x;
	bool *dev_shootList = 0;
	//cudaMalloc((void**)&dev_shootPoint, TARGRT_NUM * sizeof(bool));
	calcShootRangeList(dev_shootPoint[i], dev_theirplayer);
}

void main() {
	OurPlayer ourplayer[NUM] = {0};
	TheirPlayer theirplay[NUM] = {0};
	Point ball = {0};
	//生成射门点
	Point shootPoint[GENERATE_NUM] = {0};
	generatePoint(shootPoint, GENERATE_NUM);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
/*******************************************************/
	//获得场上信息
	getInf(ourplayer, theirplay, ball);
	//获得射门点
	Point *dev_shootPoint = 0;
	cudaMalloc((void**)&dev_shootPoint, GENERATE_NUM * sizeof(Point));
	cudaMemcpy(dev_shootPoint, shootPoint, GENERATE_NUM * sizeof(Point), cudaMemcpyHostToDevice);
	//cudaMalloc((void**)&dev_shootPoint, GENERATE_NUM * TARGRT_NUM * sizeof(bool));
	//启动CUDA运算
	//kernel <<<1, GENERATE_NUM, CONST_MEM_SIZE >>>(dev_shootPoint);
/*******************************************************/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	printf("Time for I/O : %.5f ms\n", elapseTime);

	system("pause");
}