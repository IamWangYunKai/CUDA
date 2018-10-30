#pragma comment(lib, "cuda.lib")

#define DLL_API __declspec(dllimport) 
extern "C" {
	DLL_API int addWithCuda(int *c, const int *a, const int *b, unsigned int size);
}