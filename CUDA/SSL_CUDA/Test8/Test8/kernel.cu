#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#define CAR_RADIUS 9
const unsigned int NUM = 12;
const unsigned int TARGRT_NUM = 30;
const unsigned int GENERATE_NUM = 1024;

typedef struct {
	double x;
	double y;
}Point;

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
__constant__ const Point goal_left = { 600, -60 };
__constant__ const Point goal_right = { 600, 60 };
//__constant__ Point dev_shootPoint[GENERATE_NUM];

const int CONST_MEM_SIZE = NUM * sizeof(OurPlayer) + NUM * sizeof(TheirPlayer) + 3 * sizeof(Point);

void getInf(const OurPlayer *ourplayer, const TheirPlayer *theirplayer, const Point ball) {
	cudaMemcpyToSymbol(dev_ourplayer, ourplayer, NUM * sizeof(OurPlayer));
	cudaMemcpyToSymbol(dev_theirplayer, theirplayer, NUM * sizeof(TheirPlayer));
	cudaMemcpyToSymbol(&dev_ball, &ball, sizeof(Point));
}

__device__ double CU_calcShootRange(Point source, TheirPlayer *object) {
	bool dev_shootList[TARGRT_NUM];
	double left_angle = atan2(goal_left.y - source.y, goal_left.x - source.x);
	double right_angle = atan2(goal_right.y- source.y, goal_right.x - source.x);
	double delta_angle = (left_angle - right_angle) / 30.0;
	for (int i = 0; i < TARGRT_NUM; i++) {
		dev_shootList[i] = true;//初始化
		double angle = right_angle + i * delta_angle;
		for (int j = 0; j < NUM; j++) {
			double dist = sqrt((object[j].x - source.x)*(object[j].x - source.x) + (object[j].y - source.y)*(object[j].y - source.y));
			double dir = atan2(object[j].y - source.y, object[j].x - source.x);
			double theta = asin(CAR_RADIUS / dist);
			if (angle > dir - theta || angle < dir + theta) {
				dev_shootList[i] = false;
				break;
			}
		}
	}
	//计算最大射门角
	unsigned int maxRange = 0;//最大序列
	unsigned int maxConter = 0;//最大序列头索引
	unsigned int range = 0;//当前计数器
	for (unsigned int listConter = 0; listConter < TARGRT_NUM; listConter++) {
		if (dev_shootList[listConter] == true) {
			range++;
			if (range > maxRange) {
				maxRange = range;
				maxConter = listConter - range + 1;
			}
		}
		else range = 0;
	}
	return right_angle + delta_angle * (maxConter + maxRange / 2);
}

//生成1024个射门点
void generatePoint(Point *shootPoint, unsigned int size) {
	for (int i = 0; i < size; i++) {
		shootPoint[i].x = rand() % 600;//生成0~600的整数
		shootPoint[i].y = (rand() % 900) - 450;//生成-450~450的整数
	}
}

__global__ void kernel(Point *dev_shootPoint, double *dev_max_range_list) {
	int i = threadIdx.x;
	bool *dev_shootList = 0;
	dev_max_range_list[i] = CU_calcShootRange(dev_shootPoint[i], dev_theirplayer);
}

void main() {
	OurPlayer ourplayer[NUM] = { 0 };
	TheirPlayer theirplay[NUM] = { 0 };
	Point ball = { 0 };
	double max_range_list[GENERATE_NUM] = { 0 };
	//生成射门点
	Point shootPoint[GENERATE_NUM] = { 0 };
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
	double *dev_max_range_list = 0;
	cudaMalloc((void**)&dev_shootPoint, GENERATE_NUM * sizeof(Point));
	cudaMemcpy(dev_shootPoint, shootPoint, GENERATE_NUM * sizeof(Point), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_shootPoint, shootPoint, GENERATE_NUM * sizeof(Point));
	cudaMalloc((void**)&dev_max_range_list, GENERATE_NUM * sizeof(Point));
	//启动CUDA运算
	kernel <<<1, GENERATE_NUM, CONST_MEM_SIZE >>>(dev_shootPoint, dev_max_range_list);
	cudaMemcpy(max_range_list, dev_max_range_list, GENERATE_NUM * sizeof(double), cudaMemcpyDeviceToHost);
	//释放内存
	cudaFree(dev_shootPoint);
	cudaFree(dev_max_range_list);
	/*******************************************************/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	printf("Time for I/O : %.5f ms\n", elapseTime);

	for (int i = 0; i < GENERATE_NUM; i++) {
		printf("Max shoot range: %.2lf\n", max_range_list[i]);
	}
	printf("Time for I/O : %.5f ms\n", elapseTime);
	system("pause");
}