#include "stdafx.h"
#include <iostream>

using namespace std;
const unsigned int TARGRT_NUM = 30;

void getMaxRange(bool *list, unsigned int &maxRange, unsigned int &maxConter) {
	maxRange = 0;//最大序列
	maxConter = 0;//最大序列头索引
	unsigned int range = 0;//当前计数器
	for (unsigned int listConter = 0; listConter < TARGRT_NUM; listConter++) {
		if (list[listConter] == true) {
			range++;
			if (range > maxRange) {
				maxRange = range;
				maxConter = listConter - range + 1;
			}
		}
		else range = 0;
	}
}

int main(){
	bool list[TARGRT_NUM] = {0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1};
	unsigned int maxRange = 0;
	unsigned int maxConter = 0;
	getMaxRange(list, maxRange, maxConter);
	cout << "maxRange: " << maxRange << "\tmaxConter: " << maxConter << endl;
	system("pause");
    return 0;
}

