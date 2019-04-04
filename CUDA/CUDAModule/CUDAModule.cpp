#include "CUDAModule.h"
#include <cuda.h>
#include <vector>
#include <builtin_types.h>
#include <QDebug>
#include <GDebugEngine.h>
#include <geometry.h>
#include "Semaphore.h"
extern Semaphore vision_to_cuda;
const double PI = 3.14159265359;
const double BallLineValidThreadhold = 30;
const double wTime = -0.2;
const double wShootAngle = 0.2;
const double wDist = -0.4;
const double wRefracAngle = -0.2;
const double wDeltaTime = 0.2;
const double THEIR_ROBOT_INTER_THREADHOLD = 30;

extern "C" {
    void BestPass(const Player*, const Point, rType* result);
}

CUDAModule::CUDAModule() {
    pVision = nullptr;
    thread = nullptr;
}

CUDAModule::~CUDAModule() {
}

void CUDAModule::initialize(const CVisionModule *pVision) {
    this->pVision = pVision;
    thread = new std::thread([ = ] {run();});
    thread->detach();
}

ZSS_THREAD_FUNCTION void CUDAModule::run() {
    while(true) {
        vision_to_cuda.Wait();
        ZCUDAModule::instance()->calculateBestPass();
    }
}

void CUDAModule::calculateBestPass() {
    static Player players[2 * Param::Field::MAX_PLAYER_NUM];
    static Point ball;
    CGeoPoint ballPos = pVision->Ball().Pos();
    passPoints.clear();
    chipPoints.clear();
    rType *result = new rType[12 * 16 * 128 * 2];
    //找到带球的我方车
    int minIdx = 0;
    double minDist = 1e8;
    for(int i = 0; i < Param::Field::MAX_PLAYER_NUM; i++) {
        double temp = this->pVision->OurPlayer(i + 1).Pos().dist(this->pVision->Ball().Pos());
        if(temp < minDist) {
            minIdx = i + 1;
            minDist = temp;
        }
    }

    for(int i = 0; i < Param::Field::MAX_PLAYER_NUM; i++) {
        players[i].Pos.x = this->pVision->OurPlayer(i + 1).Pos().x();
        players[i].Pos.y = this->pVision->OurPlayer(i + 1).Pos().y();
        players[i].Vel.x = this->pVision->OurPlayer(i + 1).VelX();
        players[i].Vel.y = this->pVision->OurPlayer(i + 1).VelY();
        players[i].isValid = this->pVision->OurPlayer(i + 1).Valid();
        //判断禁区和后卫
        if(players[i].isValid && (players[i].Pos.x < -460 || players[i].Pos.x > 460) && players[i].Pos.y > -140 && players[i].Pos.y < 140)
            players[i].isValid = false;

    }
    for(int i = 0; i < Param::Field::MAX_PLAYER_NUM; i++) {
        players[i + Param::Field::MAX_PLAYER_NUM].Pos.x = this->pVision->TheirPlayer(i + 1).Pos().x();
        players[i + Param::Field::MAX_PLAYER_NUM].Pos.y = this->pVision->TheirPlayer(i + 1).Pos().y();
        players[i + Param::Field::MAX_PLAYER_NUM].Vel.x = this->pVision->TheirPlayer(i + 1).VelX();
        players[i + Param::Field::MAX_PLAYER_NUM].Vel.y = this->pVision->TheirPlayer(i + 1).VelY();
        players[i + Param::Field::MAX_PLAYER_NUM].isValid = this->pVision->TheirPlayer(i + 1).Valid();
        //判断禁区和后卫
        if(players[i + Param::Field::MAX_PLAYER_NUM].isValid && (players[i + Param::Field::MAX_PLAYER_NUM].Pos.x < -460 || players[i + Param::Field::MAX_PLAYER_NUM].Pos.x > 460) && players[i + Param::Field::MAX_PLAYER_NUM].Pos.y > -140 && players[i + Param::Field::MAX_PLAYER_NUM].Pos.y < 140)
            players[i + Param::Field::MAX_PLAYER_NUM].isValid = false;

    }
    players[minIdx - 1].isValid = false;
    ball.x = this->pVision->Ball().Pos().x();
    ball.y = this->pVision->Ball().Pos().y();

    BestPass(players, ball, result);

    double max_q = -99999;
    double max_q_x, max_q_y;
    double max_q_Chip = -99999;
    double max_q_x_Chip, max_q_y_Chip;
    max_q_x = max_q_y = 9999;
    max_q_x_Chip = max_q_y_Chip = 9999;


    for(int i = 0; i < 12 * 16 * 128; i++) {
        if(result[i].interTime < 10 && result[i].interTime > 0) {
            CGeoPoint p0(result[i].interPos.x, result[i].interPos.y), p1(600, -60), p2(600, 60);
            if(!isValidPassLine(p0, ballPos)) continue;
            passPoints.push_back(result[i]);
            CVector v1 = p1 - p0,
                    v2 = p2 - p0;
            double shootAngle = fabs(Utils::Normalize(v1.dir() - v2.dir()));
            double dist = p0.dist(p1.midPoint(p2));
            v1 = ballPos - p0;
            v2 = p1.midPoint(p2) - p0;
            double refracAngle = fabs(Utils::Normalize(v1.dir() - v2.dir()));
            result[i].Q = wTime * result[i].interTime + wShootAngle * shootAngle / PI + wDist * dist / 600 + wRefracAngle * refracAngle / PI + wDeltaTime * result[i].deltaTime;

            if (result[i].Q > max_q) {
                max_q = result[i].Q;
                max_q_x = result[i].interPos.x;
                max_q_y = result[i].interPos.y;
            }
        }
    }
    for(int i = 12 * 16 * 128; i < 12 * 16 * 128 * 2; i++) {
        if(result[i].interTime < 10 && result[i].interTime > 0) {
            CGeoPoint p0(result[i].interPos.x, result[i].interPos.y), p1(600, -60), p2(600, 60);
            chipPoints.push_back(result[i]);
            CVector v1 = p1 - p0,
                    v2 = p2 - p0;
            double shootAngle = fabs(Utils::Normalize(v1.dir() - v2.dir()));
            double dist = p0.dist(p1.midPoint(p2));
            v1 = ballPos - p0;
            v2 = p1.midPoint(p2) - p0;
            double refracAngle = fabs(Utils::Normalize(v1.dir() - v2.dir()));
            result[i].Q = wTime * result[i].interTime + wShootAngle * shootAngle / PI + wDist * dist / 600 + wRefracAngle * refracAngle / PI + wDeltaTime * result[i].deltaTime;

            if (result[i].Q > max_q_Chip) {
                max_q_Chip = result[i].Q;
                max_q_x_Chip = result[i].interPos.x;
                max_q_y_Chip = result[i].interPos.y;
            }
        }
    }


    GDebugEngine::Instance()->gui_debug_x(CGeoPoint(max_q_x, max_q_y), COLOR_WHITE);
    GDebugEngine::Instance()->gui_debug_line(CGeoPoint(ball.x, ball.y), CGeoPoint(max_q_x, max_q_y), COLOR_WHITE);
    GDebugEngine::Instance()->gui_debug_line(CGeoPoint(max_q_x, max_q_y),CGeoPoint(600, 0), COLOR_WHITE);

    GDebugEngine::Instance()->gui_debug_x(CGeoPoint(max_q_x_Chip, max_q_y_Chip), COLOR_PURPLE);
    GDebugEngine::Instance()->gui_debug_line(CGeoPoint(ball.x, ball.y), CGeoPoint(max_q_x_Chip, max_q_y_Chip), COLOR_PURPLE);
    GDebugEngine::Instance()->gui_debug_line(CGeoPoint(max_q_x_Chip, max_q_y_Chip),CGeoPoint(600, 0), COLOR_PURPLE);

    for(int i = 0; i < passPoints.size(); i += 4) {
        GDebugEngine::Instance()->gui_debug_x(CGeoPoint(passPoints[i].interPos.x, passPoints[i].interPos.y), COLOR_RED);
    }

    for(int i = 0; i < chipPoints.size(); i += 4) {
        GDebugEngine::Instance()->gui_debug_x(CGeoPoint(chipPoints[i].interPos.x, chipPoints[i].interPos.y), COLOR_BLUE);
    }

    delete[] result;
}

bool CUDAModule::isValidPassLine(const CGeoPoint& p1, const CGeoPoint& p2) {
    double x1 = p1.x(),
            x2 = p2.x(),
            y1 = p1.y(),
            y2 = p2.y();
    double k, interX, interY, result, x0, y0;
    result = -9999;
    for(int i = 0; i < Param::Field::MAX_PLAYER_NUM; i++) {
        if(!pVision->TheirPlayer(i + 1).Valid())
            continue;

        CGeoPoint targetPoint = pVision->TheirPlayer(i + 1).Pos();
        x0 = targetPoint.x();
        y0 = targetPoint.y();
        if(fabs(x2 - x1) > 0.1) {
            k = (y2 - y1) / (x2 - x1);
            interX = (x0 + k * k * x1 + k * (y0 - y1)) / (1 + k * k);
            interY = k * (interX - x1) + y1;
        }
        else {
            interX = x1;
            interY = y0;
        }
        CGeoPoint interPoint(interX, interY);
        CVector direc1 = interPoint - p1,
                direc2 = interPoint - p2;
        if(direc1 * direc2 > 0) {
            CVector current2Start = p1 - targetPoint,
                    current2End = p2 - targetPoint;
            double d1 = current2Start.mod(),
                    d2 = current2End.mod();
            result = d1 > d2 ? d2 : d1;
        }
        else {
            CVector current2InterPoint = interPoint - targetPoint;
            result = current2InterPoint.mod();
        }
        if(result < THEIR_ROBOT_INTER_THREADHOLD) {
            return false;
        }
    }
    return true;
}

