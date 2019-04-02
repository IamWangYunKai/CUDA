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
	float V[N_in][N_hidden];  //���뵽���ز�Ȩֵ����
	float W[N_hidden][N_out]; //���ز㵽�������ԪȨֵ����
	float H[N_hidden];	      //���ز���Ԫ��ֵ
	float O[N_out];          //�������Ԫ��ֵ
}Net;


//gradient = f(x)(1-f(x))
float sigmoid(float x){
	return 1. / (1. + exp(-x));
}

void init(Net *net){
	srand((unsigned)time(NULL));//��ʼ���������

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
	float dealt_h[N_hidden], dealt_O[N_out];//ѵ��������Ȩ�仯��
	float out1[N_hidden], out2[N_out];

	float temp;
	//���ƽṹ�������뵽�м���Ȩֵ����
	for (int i = 0; i < N_in; i++)
		for (int j = 0; j < N_hidden; j++)
			v[i][j] = (*net).V[i][j];
	//���ƽṹ�����м�㵽������Ȩֵ����
	for (int i = 0; i < N_hidden; i++)
		for (int j = 0; j < N_out; j++)
			w[i][j] = (*net).W[i][j];
	//���ƽṹ�����ز���������ֵ
	for (int i = 0; i < N_hidden; i++)
		H[i] = (*net).H[i];
	for (int i = 0; i < N_out; i++)
		O[i] = (*net).O[i];

	float e = accuracy + 1;//ȷ���״�ѭ���ܹ�����

	for (int n = 0; e > accuracy; n++){
		e = 0;//��ʼ��eֵ
		for (int i = 0; i < N_sample; i++){
			//�����м���������
			for (int k = 0; k < N_hidden; k++){
				temp = 0;
				for (int j = 0; j < N_in; j++){
					temp += x[i][j] * v[j][k];
				}
				out1[k] = sigmoid(temp - H[k]);
			}
			//����������������
			for (int k = 0; k < N_out; k++){
				temp = 0;
				for (int j = 0; j < N_hidden; j++)
					temp += out1[j] * w[j][k];
				out2[k] = sigmoid(temp - O[k]);
			}

			//����������Ȩ�޸���,���������Ԫ�ݶ��dealt_O[]����
			for (int j = 0; j < N_out; j++)
				dealt_O[j] = out2[j] * (1 - out2[j])*(y[i][j] - out2[j]);
			//�������ز��Ȩ�޸���,�����ز���Ԫ�ݶ��dealt_h[]��
			for (int j = 0; j < N_hidden; j++){
				temp = 0;
				for (int k = 0; k < N_out; k++)
					temp += dealt_O[k] * w[j][k];
				dealt_h[j] = out1[j] * (1 - out1[j])*temp;
			}

			//�����Ȩֵ������޸�
			for (int j = 0; j < N_hidden; j++)
				for (int k = 0; k < N_out; k++)
					w[j][k] += lr * out1[j] * dealt_O[k];
			//�������ֵ�޸�
			for (int j = 0; j < N_out; j++)
				O[j] -= lr * dealt_O[j];
			//�м��Ȩֵ�����޸�
			for (int j = 0; j < N_in; j++)
				for (int k = 0; k < N_hidden; k++)
					v[j][k] += lr * x[i][j] * dealt_h[k];
			//�м����ֵ�޸�
			for (int j = 0; j < N_hidden; j++)
				H[j] -= lr * dealt_h[j];

			//�����������ԱȽϺ;��ȵĴ�С�Ӷ���ֹѭ��
			//ѵ����һ��ѵ����ɺ��ۻ����
			for (int j = 0; j < N_out; j++)
				e += (y[i][j] - out2[j]) * (y[i][j] - out2[j]);
		}
		if ((n % 10) == 0) printf("�ۼ����Ϊ��%f\n", e);
	}
	/*
	//�޸ĺ�����ز�Ȩֵ����&&���ز���ֵ
	printf("�޸ĺ�����ز�Ȩֵ����:\n");
	for (int i = 0; i < N_in; i++){
		for (int j = 0; j < N_hidden; j++)
			printf("%f	", v[i][j]);
		printf("\n");
	}
	printf("�޸ĺ�����ز���ֵ:\n");
	for (int i = 0; i < N_hidden; i++)
		printf("%f	", H[i]);
	printf("\n");
	//�޸ĺ������Ȩֵ����&&�������ֵ
	printf("�޸ĺ�������Ȩֵ����:\n");
	for (int i = 0; i < N_hidden; i++){
		for (int j = 0; j < N_out; j++)
			printf("%f	", w[i][j]);
		printf("\n");
	}
	printf("�޸ĺ���������ֵ����:\n");
	for (int i = 0; i < N_out; i++)
		printf("%f	", O[i]);
	printf("\n");
	*/

	//��������ƻؽṹ��
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
	printf("net������ѵ��������\n");
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
	//ѵ������
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
	init(&net);  //��ʼ������
	train(&net, x, y);	//ѵ������
	float result[N_out];
	forward(&net, x[4], &result[N_out]);	//���Թ���
	printf("Result: %f\n", result[N_out]);
	return 0;
}
