// callDLL.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <windows.h>  
#include <iostream>  

using namespace std;
typedef void(*FUNA)();//����ָ���DLL����ͬ�ĺ���ԭ��ָ��

int main() {
		const char* funName = "hello";
		int x(100), y(100);
		HINSTANCE hDLL = LoadLibrary("libTest.dll");
		if (hDLL != NULL){
			FUNA fp = FUNA(GetProcAddress(hDLL, "hello"));
			//��ȡ���뵽Ӧ�ó����еĺ���ָ�룬���ݷ�����ȡ��
			//FUNA fp = FUNA(GetProcAddress(hDLL,MAKEINTRESOURCE(1)));
			//����ֱ��ʹ��DLL�к������ֵ�˳���
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