// libTest.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "stdafx.h"
#include <stdio.h>
using namespace std;

#define DLL_API __declspec(dllexport) 

extern "C" {
	DLL_API void hello();
}

DLL_API void hello() {
	printf("Hello World!");
}
