// libTest.cpp : 定义 DLL 应用程序的导出函数。
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
