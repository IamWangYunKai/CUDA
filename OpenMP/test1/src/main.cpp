#include <iostream>
#include <omp.h>

using namespace std;

int main(){

#ifndef _OPENMP
	cout << "OpenMP not supporte !" << endl;
# endif

	float startTime = omp_get_wtime();

// use parallel for loop
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < 8; i++){
		cout << "This is thread: " << omp_get_thread_num() << endl;
	}

	float endTime = omp_get_wtime();
	cout << "Used time: " << endTime - startTime <<endl;

// use reductions, operator can be + - * & | ^ && ||
	const int size = 100;
	double array[size] = {0};
	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < size; i++) {
	    sum += array[i];
	}

/*
"#pragma omp parallel" will parallelly run the code in parallel region.
If your CPU's max threads number is 8, then the code will run 8 times. 
"private(tid, nthreads)" will copy data from public to private.
*/
	int tid, nthreads;
#pragma omp parallel private(tid, nthreads)
{
	tid = omp_get_thread_num();
	cout << "This is thread: " << tid << endl;
// Synchronize all threads
#pragma omp barrier
	if (0 == tid){
		nthreads = omp_get_num_threads();
		cout << "Number of threads: " << nthreads << endl;
	}
}

	getchar();
	return 0;
}