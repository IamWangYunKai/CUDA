#include "stdafx.h"
#include <windows.h>  
using namespace std;
typedef int(*FUNA)(int *c, const int *a, const int *b, const unsigned int size);
//����ָ���DLL����ͬ�ĺ���ԭ��ָ��

int main() {
	const char* funName = "addWithCuda";
	const char* dllName = "cuda.dll";
	const unsigned int size = 5;
	int a[size] = {1, 2, 3, 4, 5};
	int b[size] = {10, 20, 30, 40, 50};
	int c[size] = {0};
	HINSTANCE hDLL = LoadLibrary(dllName);
	if (hDLL != NULL){
		FUNA fp = FUNA(GetProcAddress(hDLL, funName));
		//��ȡ���뵽Ӧ�ó����еĺ���ָ�룬���ݷ�����ȡ��
		//FUNA fp = FUNA(GetProcAddress(hDLL,MAKEINTRESOURCE(1)));
		//����ֱ��ʹ��DLL�к������ֵ�˳���
		if (fp != NULL){
			fp(c, a, b, size);
			printf("  GPU Calculate:\n");
			for (int i = 0; i < size; i++)
				printf("  %d + %d = %d\n", a[i], b[i], c[i]);
		}
		else{
			printf("Cannot Find Function: %s", funName);
		}
		FreeLibrary(hDLL);
	}
	else{
		printf("Cannot Find dll: %s", dllName);
	}
	getchar();
	return 1;
}