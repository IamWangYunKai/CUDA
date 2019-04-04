#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_functions.h>
#include "CMmotion.cu"
#include <stdio.h>

#define FRAME_PERIOD (1 / 60.0)
#define ZERO_NUM (1e-8)
#define A_FACTOR (1.5)
#define OUR_MAX_ACC (450)
#define OUR_MAX_DEC (450)
#define OUR_MAX_VEL (300)
#define THEIR_MAX_ACC (500)
#define THEIR_MAX_DEC (500)
#define THEIR_MAC_VEL (350)
#define PI (3.14159265359)
#define G (9.8)
#define SQRT_2 (1.414)
#define TIME_FOR_OUR (0)
#define TIME_FOR_OUR_BOTH_KEEP (-0.2)
#define TIME_FOR_THEIR_BOTH_KEEP (-0.4)
#define TIME_FOR_THEIR (-0.6)
#define TIME_FOR_JUDGE_HOLDING (0.5)

#define FRICTION (87)
#define PLAYER_CENTER_TO_BALL_CENTER (60)
#define MAX_PLAYER_NUM (12)
#define THREAD_NUM (128)
#define BLOCK_X (16)
#define BLOCK_Y (MAX_PLAYER_NUM * 2)
#define MAX_BALL_SPEED (650)
#define MIN_BALL_SPEED (50)
#define BALL_SPEED_UNIT ((MAX_BALL_SPEED - MIN_BALL_SPEED) / BLOCK_X)
#define MIN_DELTA_TIME (0.2)


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

__device__ bool IsInField(Point p) {
    return (p.x > -600 && p.x < 600 && p.y < 450 && p.y > -450);
}

__device__ bool IsInPenalty(Point p) {
    return (p.x < -480 && p.x > -600 && p.y > -120 && p.y < 120)
            || (p.x > 480 && p.x < 600 && p.y > -120 && p.y < 120);
}

__device__ bool predictedInterTime(Point mePoint, Point ballPoint, Vector meVel, Vector ballVel, Point* interceptPoint, double* interTime, double responseTime) {
    if(sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) < 40){
        *interceptPoint = ballPoint;//截球点
        *interTime = predictedTime(mePoint.x, mePoint.y, interceptPoint->x, interceptPoint->y, meVel.x, meVel.y);//截球时间
        return true;
    }
    const double ballAcc = FRICTION / 2;//球减速度
    double ballArriveTime = 0;
    double meArriveTime = 9999;
    const double stepTime = 0.1;//最少帧数
    double testBallLength = 0;//球移动距离
    Point testPoint = ballPoint;
    double testVel = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
    double max_time = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) / ballAcc;

    bool canInter = true;
    for (ballArriveTime = 0; ballArriveTime < max_time; ballArriveTime += stepTime ) {
        testVel = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) - ballAcc*ballArriveTime;//v_0-at
        testBallLength = PLAYER_CENTER_TO_BALL_CENTER + (sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) + testVel)*ballArriveTime / 2;//梯形法计算球移动距离
        Vector direc;
        direc.x = testBallLength * ballVel.x / sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
        direc.y = testBallLength * ballVel.y / sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
        testPoint.x = ballPoint.x + direc.x;
        testPoint.y = ballPoint.y + direc.y;
        meArriveTime = predictedTime(mePoint.x, mePoint.y, testPoint.x, testPoint.y, meVel.x, meVel.y);//我到截球点的时间
        if(meArriveTime < 0.15) meArriveTime = 0;
        if(IsInPenalty(testPoint)) continue;
        if (!IsInField(testPoint)) {
            canInter = false;
            break;
        }
        if(meArriveTime + responseTime < ballArriveTime) break;
    }
    if(!canInter || ballArriveTime >= max_time) {
        interceptPoint->x = 9999;
        interceptPoint->y = 9999;
        *interTime = 9999;
        return false;
    }
    *interceptPoint = testPoint;//截球点
    *interTime = predictedTime(mePoint.x, mePoint.y, interceptPoint->x, interceptPoint->y, meVel.x, meVel.y);//截球时间
    return true;
}

__device__ bool predictedChipInterTime(Point mePoint, Point ballPoint, Vector meVel, Vector ballVel, Point* interceptPoint, double* interTime, double responseTime) {
    double chipVel = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
    double meArriveTime = 9999;
    double ballAcc = FRICTION / 2.0;//球减速度
    double stepTime = 0.1;
    double testBallLength = 0;//球移动距离
    Point testPoint = ballPoint;

    double factor_1 = 0.2;
    double factor_2 = 0.1;

    double time_1 = SQRT_2*chipVel/100.0/G;
    double time_2 = SQRT_2*chipVel*factor_1/100.0/G;
    double length_1 = (chipVel*chipVel)/100/G;
    double length_2 = (factor_1*chipVel*factor_1*chipVel)/100/G;
    double moveVel = chipVel / SQRT_2 * factor_2;

    double max_time = SQRT_2 * chipVel / 100 / G + SQRT_2 * 0.2 * chipVel / 100 / G  + chipVel / (SQRT_2 * ballAcc);
//    printf("%f\n", ballAcc);
    bool canInter = true;
    double ballArriveTime = time_1 + time_2;

    while (ballArriveTime < max_time) {
        Vector direc;
        testBallLength = length_1 + length_2 + (moveVel * ballArriveTime - 0.5*ballAcc*ballArriveTime*ballArriveTime);

        direc.x = testBallLength * ballVel.x / sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
        direc.y = testBallLength * ballVel.y / sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
        testPoint.x = ballPoint.x + direc.x;
        testPoint.y = ballPoint.y + direc.y;
        meArriveTime = predictedTime(mePoint.x, mePoint.y, testPoint.x, testPoint.y, meVel.x, meVel.y);//我到截球点的时间
        if(meArriveTime < 0.10) meArriveTime = 0;

        if(IsInPenalty(testPoint)) {
            ballArriveTime += stepTime;
            continue;
        }
        if (!IsInField(testPoint)) {
            canInter = false;
            break;
        }
        if(meArriveTime + responseTime < ballArriveTime) break;
        ballArriveTime += stepTime;
    }

    if(!canInter || ballArriveTime >= max_time) {
        interceptPoint->x = 9999;
        interceptPoint->y = 9999;
        *interTime = 9999;
        return false;
    }

    *interceptPoint = testPoint;//截球点
    *interTime = predictedTime(mePoint.x, mePoint.y, interceptPoint->x, interceptPoint->y, meVel.x, meVel.y);//截球时间
    return true;
}

__device__ bool predictedTheirInterTime(Point mePoint, Point ballPoint, Vector meVel, Vector ballVel, Point* interceptPoint, double* interTime, double responseTime) {
    if(sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) < 40){
        *interceptPoint = ballPoint;//截球点
        *interTime = predictedTheirTime(mePoint.x, mePoint.y, interceptPoint->x, interceptPoint->y, meVel.x, meVel.y);//截球时间
        return true;
    }
    const double ballAcc = FRICTION / 2;//球减速度
    double ballArriveTime = 0;
    double meArriveTime = 9999;
    const double stepTime = 0.1;//最少帧数
    double testBallLength = 0;//球移动距离
    Point testPoint = ballPoint;
    double testVel = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
    double max_time = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) / ballAcc;

    bool canInter = true;
    for (ballArriveTime = 0; ballArriveTime < max_time; ballArriveTime += stepTime ) {
        testVel = sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) - ballAcc*ballArriveTime;//v_0-at
        testBallLength = PLAYER_CENTER_TO_BALL_CENTER + (sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y) + testVel)*ballArriveTime / 2;//梯形法计算球移动距离
        Vector direc;
        direc.x = testBallLength * ballVel.x / sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
        direc.y = testBallLength * ballVel.y / sqrt(ballVel.x * ballVel.x + ballVel.y * ballVel.y);
        testPoint.x = ballPoint.x + direc.x;
        testPoint.y = ballPoint.y + direc.y;
        meArriveTime = predictedTheirTime(mePoint.x, mePoint.y, testPoint.x, testPoint.y, meVel.x, meVel.y);//我到截球点的时间
        if(meArriveTime < 0.15) meArriveTime = 0;
        if(IsInPenalty(testPoint)) continue;
        if (!IsInField(testPoint)) {
            canInter = false;
            break;
        }
        if(meArriveTime + responseTime < ballArriveTime) break;
    }
    if(!canInter) {
        interceptPoint->x = 9999;
        interceptPoint->y = 9999;
        *interTime = 9999;
        return false;
    }
    *interceptPoint = testPoint;//截球点
    *interTime = predictedTime(mePoint.x, mePoint.y, interceptPoint->x, interceptPoint->y, meVel.x, meVel.y);//截球时间
    return true;
}

__global__ void calculateAllInterInfo(Player* players, Point* ballPos, rType* bestPass) {
    int angleIndex = threadIdx.x;
    int speedIndex = blockIdx.x;
    int playerNum =  blockIdx.y;

    Vector ballVel;
    ballVel.x = (speedIndex * BALL_SPEED_UNIT + MIN_BALL_SPEED) * cospi(2*PI* angleIndex / THREAD_NUM);
    ballVel.y = (speedIndex * BALL_SPEED_UNIT + MIN_BALL_SPEED) * sinpi(2*PI* angleIndex / THREAD_NUM);


    double interTime;
    Point interPoint;

    interTime = 9999;
    interPoint.x = 9999;
    interPoint.y = 9999;

    if( players[playerNum].isValid && playerNum < 12)
         predictedInterTime(players[playerNum].Pos, *ballPos, players[playerNum].Vel, ballVel, &interPoint, &interTime, 0);
    else if(players[playerNum].isValid)
         predictedTheirInterTime(players[playerNum].Pos, *ballPos, players[playerNum].Vel, ballVel, &interPoint, &interTime, 0);
//    if(interTime > 0 && playerNum - 12 < 10) {
//        printf("%d : %f \n", playerNum - 12, interTime);
//    }

    int offset = blockIdx.y + gridDim.y * (threadIdx.x + blockIdx.x * blockDim.x);
    bestPass[offset].interPos = interPoint;
    bestPass[offset].interTime = interTime;
    bestPass[offset].playerIndex = playerNum;
//    /***************** chip *******************/

    if( players[playerNum].isValid && playerNum < 12)
         predictedChipInterTime(players[playerNum].Pos, *ballPos, players[playerNum].Vel, ballVel, &interPoint, &interTime, 0);
      else if(players[playerNum].isValid)
         predictedChipInterTime(players[playerNum].Pos, *ballPos, players[playerNum].Vel, ballVel, &interPoint, &interTime, 0);


    offset += BLOCK_X * BLOCK_Y * THREAD_NUM;
    bestPass[offset].interPos = interPoint;
    bestPass[offset].interTime = interTime;
    bestPass[offset].playerIndex = playerNum;
    __syncthreads();
}

__global__ void getBest(rType* passPoints) {
    __shared__ rType iP[BLOCK_Y];
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int playerNum = threadIdx.x;
    iP[playerNum] = passPoints[blockId * blockDim.x + playerNum];
    __syncthreads();
    bool even = true;
    for(int i = 0; i < blockDim.x; i++) {
        if(playerNum < blockDim.x - 1 && even && iP[playerNum].interTime > iP[playerNum + 1].interTime) {
            rType temp;
            temp = iP[playerNum + 1];
            iP[playerNum + 1] = iP[playerNum];
            iP[playerNum] = temp;
        }
        else if(playerNum > 0 && !even && iP[playerNum].interTime < iP[playerNum - 1].interTime) {
            rType temp;
            temp = iP[playerNum];
            iP[playerNum] = iP[playerNum - 1];
            iP[playerNum - 1] = temp;
        }
        even = !even;
        __syncthreads();
    }
    passPoints[blockId * blockDim.x + playerNum] = iP[playerNum];

    /************************/
    __shared__ rType iP2[BLOCK_Y];
    iP2[playerNum] = passPoints[blockId * blockDim.x + playerNum + BLOCK_X * BLOCK_Y * THREAD_NUM];
    __syncthreads();
    even = true;
    for(int i = 0; i < blockDim.x; i++) {
        if(playerNum < blockDim.x - 1 && even && iP2[playerNum].interTime > iP2[playerNum + 1].interTime) {
            rType temp;
            temp = iP2[playerNum + 1];
            iP2[playerNum + 1] = iP2[playerNum];
            iP2[playerNum] = temp;
        }
        else if(playerNum > 0 && !even && iP2[playerNum].interTime < iP2[playerNum - 1].interTime) {
            rType temp;
            temp = iP2[playerNum];
            iP2[playerNum] = iP2[playerNum - 1];
            iP2[playerNum - 1] = temp;
        }
        even = !even;
        __syncthreads();
    }
    passPoints[blockId * blockDim.x + playerNum + BLOCK_X * BLOCK_Y * THREAD_NUM] = iP2[playerNum];
    __syncthreads();
}


extern "C" void BestPass(const Player* players, const Point ball, rType* result) {
    Player *dev_players;
    Point *dev_ball;
    rType  *dev_bestPass, *bestPass;

    cudaMalloc((void**)&dev_players, 2 * MAX_PLAYER_NUM * sizeof(Player));
    cudaMalloc((void**)&dev_ball, sizeof(Point));
    cudaMalloc((void**)&dev_bestPass, 2 * BLOCK_X * BLOCK_Y * THREAD_NUM * sizeof(rType));

    cudaMemcpy(dev_players, players, 2 * MAX_PLAYER_NUM * sizeof(Player), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ball, &ball, sizeof(Point), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 bolcks(BLOCK_X, BLOCK_Y);
    calculateAllInterInfo <<< bolcks, THREAD_NUM >>> (dev_players, dev_ball, dev_bestPass);

    dim3 blocks2(BLOCK_X, THREAD_NUM);
    getBest<<< blocks2, BLOCK_Y >>> (dev_bestPass);

    bestPass = new rType[2 * BLOCK_X * BLOCK_Y * THREAD_NUM];
    cudaMemcpy(bestPass, dev_bestPass, 2 * BLOCK_X * BLOCK_Y * THREAD_NUM * sizeof(rType), cudaMemcpyDeviceToHost);

    cudaError_t cudaStatus = cudaGetLastError();
//    printf("%d %d\n", cudaStatus, cudaSuccess);
    if (cudaStatus != cudaSuccess){
        printf("CUDA ERROR: %d\n", (int)cudaStatus);
        printf("Error Name: %s\n", cudaGetErrorName(cudaStatus));
        printf("Description: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %.5f ms\n", milliseconds);

    rType defaultPlayer;
    defaultPlayer.dir = 9999;
    defaultPlayer.interPos.x = 9999;
    defaultPlayer.interPos.y = 9999;
    defaultPlayer.interTime = 9999;
    defaultPlayer.Vel = 9999;
    defaultPlayer.deltaTime = -9999;
    for(int i = 0; i < BLOCK_X * BLOCK_Y * THREAD_NUM; i += BLOCK_Y) {
        int playerNum = 0;
        for(int j = 0; j < MAX_PLAYER_NUM; j++) {
            if(bestPass[i + j].playerIndex > 11) {
                while(playerNum < MAX_PLAYER_NUM) {
                    result[i / 2 + playerNum] = defaultPlayer;
                    playerNum++;
                }
                for(int k = 0; k < j; k++) {
                    result[i / 2 + k].deltaTime = bestPass[i + j].interTime - result[i / 2 + k].interTime;
                    if(result[i / 2 + k].deltaTime < MIN_DELTA_TIME)
                        result[i / 2 + k] = defaultPlayer;
                }
                break;
            }
            else {
                result[i / 2 + playerNum] = bestPass[i + j];
                playerNum++;
            }
        }
    }
    for(int i = BLOCK_X * BLOCK_Y * THREAD_NUM; i < 2 * BLOCK_X * BLOCK_Y * THREAD_NUM; i += BLOCK_Y) {
        int playerNum = 0;

        for(int j = 0; j < MAX_PLAYER_NUM; j++) {

            if(bestPass[i + j].playerIndex > 11) {
                while(playerNum < MAX_PLAYER_NUM) {
                    result[i / 2 + playerNum] = defaultPlayer;
                    playerNum++;
                }
                for(int k = 0; k < j; k++) {
                    result[i / 2 + k].deltaTime = bestPass[i + j].interTime - result[i / 2 + k].interTime;
                    if(result[i / 2 + k].deltaTime < MIN_DELTA_TIME)
                        result[i / 2 + k] = defaultPlayer;
                }
                break;
            }
            else {
                result[i / 2 + playerNum] = bestPass[i + j];
                playerNum++;
            }
        }
    }
    delete[] bestPass;
    cudaFree(dev_players);
    cudaFree(dev_ball);
    cudaFree(dev_bestPass);
}
