// callDLL.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <windows.h>  
#include <iostream>  

using namespace std;
typedef void(*FUNA)();//定义指向和DLL中相同的函数原型指针

int main() {
		const char* funName = "hello";
		int x(100), y(100);
		HINSTANCE hDLL = LoadLibrary("libTest.dll");
		if (hDLL != NULL){
			FUNA fp = FUNA(GetProcAddress(hDLL, "hello"));
			//获取导入到应用程序中的函数指针，根据方法名取得
			//FUNA fp = FUNA(GetProcAddress(hDLL,MAKEINTRESOURCE(1)));
			//根据直接使用DLL中函数出现的顺序号
			if (fp != NULL){
				fp();
			}
			else{
				cout << "Cannot Find Function : " << funName << endl;
			}
			FreeLibrary(hDLL);
		}else{
			cout << "Cannot Find dll : " << endl;
		}
		getchar();
		return 1;
	}