#include <iostream>
//当需要调用OpenMP的API时需要引用的库
#include <omp.h>
#define MAX_THREADS 8
using namespace std;

int main(){
//检查是否支持OpenMP
#ifndef _OPENMP
	cout << "OpenMP not supporte !" << endl;
# endif
//设置并行的线程数，如不设置则默认是CPU最大线程数
	omp_set_num_threads(MAX_THREADS) ;
//获取程序运行时间，单位秒
	double startTime = omp_get_wtime();
//使for循环并行计算，追加num_threads(n)可以设定线程数
//"schedule(调度类型，迭代次数)"可以实现负载平衡,调度类型有
//1.static 每次给线程分配固定迭代数 2.dynamic 动态分配，不指定迭代数则默认为1
//3.guided 启发式调度，先分配多，后分配少
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < 10*MAX_THREADS; i++){
		//获取当前线程的ID号
		cout << "This is thread: " << omp_get_thread_num() << endl;
	}
	double endTime = omp_get_wtime();
	cout << "Used time: " << endTime - startTime <<endl;

	const int size = 1024;
	double sum = 0;
//使用reduction迭加,运算可以是 + - * & | ^ && ||
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < size; i++) {
	    sum += 1;
	}
	cout << "Sum: " << sum << endl;

	int tid, nthreads;
//"#pragma omp parallel"构建了一个并行域，代码会运行MAX_THREADS次
//"private(变量1,变量2，···)"会将公有变量拷贝为每个线程的私有变量
#pragma omp parallel private(tid, nthreads)
{
	tid = omp_get_thread_num();
	cout << "This is thread: " << tid << endl;
//"barrier"用来线程同步，当有线程未完成时，其他线程不能执行后面代码
#pragma omp barrier
	if (0 == tid){
		//获取正在运行的线程数
		nthreads = omp_get_num_threads();
		cout << "Number of threads: " << nthreads << endl;
	}
}
//并行执行不同的代码块
#pragma omp parallel sections
{
#pragma omp section
    cout << "Section 1 tid: " << omp_get_thread_num() << endl;
#pragma omp section
    cout << "Section 2 tid: " << omp_get_thread_num() << endl;
#pragma omp section
    cout << "Section 3 tid: " << omp_get_thread_num() << endl;
#pragma omp section
    cout << "Section 4 tid: " << omp_get_thread_num() << endl;
}
	return 0;
}