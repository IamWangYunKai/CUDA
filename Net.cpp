#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_in	 5
#define N_out    1
#define N_sample 12
#define N_hidden 32

const float lr = 0.01;
const float accuracy = 0.01;

typedef struct{
	float V[N_in][N_hidden];  //输入到隐藏层权值矩阵
	float W[N_hidden][N_out]; //隐藏层到输出层神经元权值矩阵
	float H[N_hidden];	      //隐藏层神经元阙值
	float O[N_out];          //输出层神经元阙值
}Net;


//gradient = f(x)(1-f(x))
float sigmoid(float x){
	return 1. / (1. + exp(-x));
}

void init(Net *net){
	srand((unsigned)time(NULL));//初始化随机种子

	for (int i = 0; i < N_in; i++)
		for (int j = 0; j < N_hidden; j++)
			(*net).V[i][j] = rand() / (RAND_MAX + 1.0) * sqrt(6./(N_in+ N_hidden));
	for (int i = 0; i < N_hidden; i++)
		for (int j = 0; j < N_out; j++)
			(*net).W[i][j] = rand() / (RAND_MAX + 1.0) * sqrt(6. / (N_out + N_hidden));
	for (int i = 0; i < N_hidden; i++)
		(*net).H[i] = rand() / (RAND_MAX + 1.0);
	for (int i = 0; i < N_out; i++)
		(*net).O[i] = rand() / (RAND_MAX + 1.0);
}


void train(Net *net, float x[N_sample][N_in], float y[N_sample][N_out]){
	float v[N_in][N_hidden], w[N_hidden][N_out];
	float H[N_hidden], O[N_out];
	float dealt_h[N_hidden], dealt_O[N_out];//训练过程中权变化量
	float out1[N_hidden], out2[N_out];

	float temp;
	//复制结构体中输入到中间层的权值矩阵
	for (int i = 0; i < N_in; i++)
		for (int j = 0; j < N_hidden; j++)
			v[i][j] = (*net).V[i][j];
	//复制结构体中中间层到输出层的权值矩阵
	for (int i = 0; i < N_hidden; i++)
		for (int j = 0; j < N_out; j++)
			w[i][j] = (*net).W[i][j];
	//复制结构体隐藏层和输出层阙值
	for (int i = 0; i < N_hidden; i++)
		H[i] = (*net).H[i];
	for (int i = 0; i < N_out; i++)
		O[i] = (*net).O[i];

	float e = accuracy + 1;//确保首次循环能够进行

	for (int n = 0; e > accuracy; n++){
		e = 0;//初始化e值
		for (int i = 0; i < N_sample; i++){
			//计算中间层输出向量
			for (int k = 0; k < N_hidden; k++){
				temp = 0;
				for (int j = 0; j < N_in; j++){
					temp += x[i][j] * v[j][k];
				}
				out1[k] = sigmoid(temp - H[k]);
			}
			//计算输出层输出向量
			for (int k = 0; k < N_out; k++){
				temp = 0;
				for (int j = 0; j < N_hidden; j++)
					temp += out1[j] * w[j][k];
				out2[k] = sigmoid(temp - O[k]);
			}

			//计算输出层的权修改量,即输出层神经元梯度项（dealt_O[]）、
			for (int j = 0; j < N_out; j++)
				dealt_O[j] = out2[j] * (1 - out2[j])*(y[i][j] - out2[j]);
			//计算隐藏层的权修改量,即隐藏层神经元梯度项（dealt_h[]）
			for (int j = 0; j < N_hidden; j++){
				temp = 0;
				for (int k = 0; k < N_out; k++)
					temp += dealt_O[k] * w[j][k];
				dealt_h[j] = out1[j] * (1 - out1[j])*temp;
			}

			//输出层权值矩阵的修改
			for (int j = 0; j < N_hidden; j++)
				for (int k = 0; k < N_out; k++)
					w[j][k] += lr * out1[j] * dealt_O[k];
			//输出层阙值修改
			for (int j = 0; j < N_out; j++)
				O[j] -= lr * dealt_O[j];
			//中间层权值矩阵修改
			for (int j = 0; j < N_in; j++)
				for (int k = 0; k < N_hidden; k++)
					v[j][k] += lr * x[i][j] * dealt_h[k];
			//中间层阙值修改
			for (int j = 0; j < N_hidden; j++)
				H[j] -= lr * dealt_h[j];

			//计算输出误差以比较和精度的大小从而终止循环
			//训练集一次训练完成后累积误差
			for (int j = 0; j < N_out; j++)
				e += (y[i][j] - out2[j]) * (y[i][j] - out2[j]);
		}
		if ((n % 10) == 0) printf("累计误差为：%f\n", e);
	}
	/*
	//修改后的隐藏层权值矩阵&&隐藏层阙值
	printf("修改后的隐藏层权值矩阵:\n");
	for (int i = 0; i < N_in; i++){
		for (int j = 0; j < N_hidden; j++)
			printf("%f	", v[i][j]);
		printf("\n");
	}
	printf("修改后的隐藏层阙值:\n");
	for (int i = 0; i < N_hidden; i++)
		printf("%f	", H[i]);
	printf("\n");
	//修改后输出层权值矩阵&&输出层阙值
	printf("修改后的输出层权值矩阵:\n");
	for (int i = 0; i < N_hidden; i++){
		for (int j = 0; j < N_out; j++)
			printf("%f	", w[i][j]);
		printf("\n");
	}
	printf("修改后的输出层阙值数组:\n");
	for (int i = 0; i < N_out; i++)
		printf("%f	", O[i]);
	printf("\n");
	*/

	//将结果复制回结构体
	for (int i = 0; i < N_in; i++)
		for (int j = 0; j < N_hidden; j++)
			(*net).V[i][j] = v[i][j];
	for (int i = 0; i < N_hidden; i++)
		for (int j = 0; j < N_out; j++)
			(*net).W[i][j] = w[i][j];
	for (int i = 0; i < N_hidden; i++)
		(*net).H[i] = H[i];
	for (int i = 0; i < N_out; i++)
		(*net).O[i] = O[i];
	printf("net神经网络训练结束！\n");
}

void forward(Net *net, float input[N_in], float *output){
	float out1[N_hidden] = {0};

	for (int i = 0; i < N_hidden; i++) {
		float temp = 0;
		for (int j = 0; j < N_in; j++)
			temp += input[j] * (*net).V[j][i];
		out1[i] = sigmoid(temp - (*net).H[i]);
	}

	for (int i = 0; i < N_out; i++) {
		float temp = 0;
		for (int j = 0; j < N_hidden; j++)
			temp += out1[j] * (*net).W[j][i];
		output[i] = sigmoid(temp - (*net).O[i]);
	}
}

int main(){
	//训练样本
	float x[N_sample][N_in] = {
		{ 0.4,0.3,0.8 },
		{ 0.8,0.6,0.2 },
		{ 0.3,0.7,0.3 },
		{ 0.2,0.9,0.7 },
		{ 0.9,0.5,0.6 },
		{ 0.2,0.1,0.4 },
		{ 0.6,0.8,0.1 },
		{ 0.5,0.7,0.9 },
		{ 0.5,0.4,0.4 },
		{ 0.45,0.4,0.6 },
		{ 0.4,0.55,0.6 },
		{ 0.5,0.6,0.45 },
	};

	float y[N_sample][N_out] = { { 1 },{ 1 },{ 0 },{ 0 },{ 1 },{ 1},{ 0 },{ 0 },{ 1 },{ 1 },{ 0 },{ 0 } };

	Net net;
	init(&net);  //初始化过程
	train(&net, x, y);	//训练过程
	float result[N_out];
	forward(&net, x[4], &result[N_out]);	//测试过程
	printf("Result: %f\n", result[N_out]);
	return 0;
}
