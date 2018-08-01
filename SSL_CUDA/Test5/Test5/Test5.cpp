#include "stdafx.h"
#include <iostream>
#include <math.h>

using namespace std;

typedef struct {
	double x;
	double y;
}Point;

//分别传入当前点、射门点、阻挡点，若能射门返回true，否则返回false
bool calcShootRange(Point source, Point target, Point object) {
	double A = target.y - source.y;
	double B = source.x - target.x;
	double C = target.x * source.y - source.x * target.y;
	double D = sqrt(A*A + B*B);
	if (D < 0.01) return false;
	double dist = abs(A*object.x + B*object.y + C) / D;
	cout << "Dist:" << dist << endl;
	if (dist > 9.0) return true;
	return false;
}

int main() {
	Point source = { 0.0, 0.0 };
	//Point target = { -100, 300 };
	Point target[30];
	for (int i = 0; i < 30; i++) {
		target[i].x = -150 + 4 * i;
		target[i].y = 300;
	}
	Point object = { -58, 200 };
	//calcShootRange(source, target, object);
	for (int i = 0; i < 30; i++)
		calcShootRange(source, target[i], object);
	system("pause");
	return 0;
}
