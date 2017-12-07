#include "stdafx.h"
#include "cuda.h"
using namespace std;

int main() {
	const unsigned int size = 5;
	int a[size] = { 1, 2, 3, 4, 5 };
	int b[size] = { 10, 20, 30, 40, 50 };
	int c[size] = { 0 };
	addWithCuda(c, a, b, size);
	printf("  GPU Calculate:\n");
	for (int i = 0; i < size; i++){
		printf("  %d + %d = %d\n", a[i], b[i], c[i]);
	}
	getchar();
	return 1;
}