#include <iostream>
//#include <omp.h>
using namespace std;

int main(){

# ifdef _OPENMP
	cout << "Compiled by an OpenMP compliant implementation." << endl;
# endif

#pragma omp parallel for
	for (int i = 0; i < 8; i++){
		cout << "This is " << i << endl;
	}
	getchar();
	return 0;
}