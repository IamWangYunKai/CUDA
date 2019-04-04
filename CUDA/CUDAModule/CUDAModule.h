/**************************************
* CUDA C acceleration module
* Author: Wang Yun Kai
* Created Date: 2019/3/17
**************************************/
#ifndef __CUDA_MODULE_H__
#define __CUDA_MODULE_H__

#include "singleton.hpp"
#include "VisionModule.h"
#include <MultiThread.h>
#include "VisionModule.h"

typedef struct {
    double x, y;
} Vector;

typedef struct {
    double x, y;
} Point;

typedef struct {
    Point Pos;
    Vector Vel;
    bool isValid;
} Player;

typedef struct {
    Point interPos;
    double interTime;
    double Vel;
    float dir;
    int playerIndex;
    double deltaTime;
    double Q;
} rType;

class CUDAModule{
public:
    CUDAModule();
    ~CUDAModule();
    void initialize(const CVisionModule *);
    ZSS_THREAD_FUNCTION void run();
    std::vector<rType> getPassPoints(void) const { return passPoints; }
    std::vector<rType> getChipPoints(void) const { return chipPoints; }
private:
    const CVisionModule* pVision;
    std::thread *thread;
    std::vector<rType> passPoints;
    std::vector<rType> chipPoints;
    void calculateBestPass();
    bool isValidPassLine(const CGeoPoint& p1, const CGeoPoint& p2);
};
typedef Singleton<CUDAModule> ZCUDAModule;
#endif //__CUDA_MODULE_H__
