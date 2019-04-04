#include <cuda.h>
//#include <builtin_types.h>
#include <math_functions.h>

#define FRAME_PERIOD (1 / 60.0)

extern "C"{
__device__ double predictedTime(double x0, double y0, double x1, double y1, double vx, double vy);
__device__ void compute_motion_1d(double x0, double v0, double v1,
    double a_max, double d_max, double v_max, double a_factor,
    double &traj_accel, double &traj_time, double &traj_time_acc, double &traj_time_dec, double &traj_time_flat);
__device__ bool finite(double a);
}


__device__ bool finite(double a) {
    return fabs(a) < 9999;
}


__device__ void compute_motion_1d(double x0, double v0, double v1,
    double a_max, double d_max, double v_max, double a_factor,
    double &traj_accel, double &traj_time, double &traj_time_acc, double &traj_time_dec, double &traj_time_flat)
{
    if (x0 == 0 && v0 == v1) {
        traj_accel = 0;
        traj_time_acc = traj_time_dec = 0;
        return;
    }

    if(!finite(x0) || !finite(v0) || !finite(v1)) {
        traj_accel = 0;
        traj_time_acc = traj_time_dec = 0;
        return;
    }

    a_max /= a_factor;
    d_max /= a_factor;

    double accel_time_to_v1 = fabs(v1 - v0) / a_max;                                                  // 最大加速度加速到末速度的时间
    double accel_dist_to_v1 = fabs((v1 + v0) / 2.0) * accel_time_to_v1;                               // 单一加速到末速度时的位移
    double decel_time_to_v1 = fabs(v0 - v1) / d_max;                                                  // 最大减速度减速到末速度的时间
    double decel_dist_to_v1 = fabs((v0 + v1) / 2.0) * decel_time_to_v1;                               // 单一减速到末速度时的位移

    // 这个时间很关键，设得较大则定位精度将大大降低 by qxz
    double period = 1 / 40.0; // 一段很小的时间，处理运动到目标点附近时加速度，稳定到点，防止超调

    // 计算时间部分

    // 从x0运动到零点
    // 初速和目标点反向 或 初速大于目标速来不及减速到目标速
    // 全程减速
    if (v0 * x0 > 0 || (fabs(v0) > fabs(v1) && decel_dist_to_v1 > fabs(x0))) {
        // 停下后到达的时间 + 停下所用时间
        double time_to_stop = fabs(v0) / (d_max);                                                       // 停下时间
        double x_to_stop = v0 * v0 / (2.0 * d_max);                                                   // 停止时运动距离

        compute_motion_1d(x0 +  copysign(x_to_stop, v0), 0, v1, a_max * a_factor, d_max * a_factor,
            v_max, a_factor, traj_accel, traj_time, traj_time_acc, traj_time_dec, traj_time_flat);    // 递归运算直到跳出这一条件
        traj_time += time_to_stop;                                                                    // 加上路径规划时间
        traj_time /= 1.25;
        traj_accel = 0;

        return;
    }

    // 初速和目标点同向
    if (fabs(v0) > fabs(v1)) {                                                                          // 初速度大于目标速，但可以减速到目标速 先加速再减速
        traj_time_acc = (sqrt((d_max * v0 * v0 + a_max * (v1 * v1 + 2 * d_max * fabs(x0))) / (a_max + d_max)) - fabs(v0)) / a_max;

        if (traj_time_acc < 0.0)
            traj_time_acc = 0;
        traj_time_dec = ((fabs(v0) - fabs(v1)) + a_max * traj_time_acc) / d_max;
    }

    else if (accel_dist_to_v1 > fabs(x0)) {                                                             // 初速度小于目标速，且不可加速到目标速 全程加速
        traj_time_acc = (sqrt(v0 * v0 + 2 * a_max * fabs(x0)) - fabs(v0)) / a_max;
        traj_time_dec = 0.0;
    }

    else {                                                                                              // 初速度小于目标速，且可以加速到目标速 先加速再减速
        traj_time_acc = (sqrt((d_max * v0 * v0 + a_max * (v1 * v1 + 2 * d_max * fabs(x0))) / (a_max + d_max)) - fabs(v0)) / a_max;
        if (traj_time_acc < 0.0)
            traj_time_acc = 0;
        traj_time_dec = ((fabs(v0) - fabs(v1)) + a_max * traj_time_acc) / d_max;
    }

    // 计算所得车速可能超过车最大速度，会有一段匀速运动

    if (traj_time_acc * a_max + fabs(v0) > v_max) {                                                     // 匀速运动的时间
        double dist_without_flat = (v_max * v_max - v0 * v0) / (2 * a_max) + (v_max * v_max - v1 * v1) / (2 * d_max);
        traj_time_flat = (fabs(x0) - dist_without_flat) / v_max;
    }
    else {
        traj_time_flat = 0;
    }

    // 分配加速度部分

    double a_to_v1_at_x0 = fabs(v0 * v0 - v1 * v1) / (2 * fabs(x0));
    double t_to_v1_at_x0 = (-fabs(v0) + sqrt(v0 * v0 + 2 * fabs(a_to_v1_at_x0) * fabs(x0))) / fabs(a_to_v1_at_x0);
    if (t_to_v1_at_x0 < period) {
        traj_accel = - copysign(a_to_v1_at_x0, v0);
        return;
    }

    if (FRAME_PERIOD * a_max + fabs(v0) > v_max && traj_time_flat > period) {                           // 匀速运动阶段
        traj_time = traj_time_flat + traj_time_dec;
        traj_accel = 0;
    }
    else if (traj_time_acc < period && traj_time_dec == 0.0) {                                          // 加速到点
        traj_time = traj_time_acc;
        traj_accel =  copysign(a_max * a_factor, -x0);
    }
    else if (traj_time_acc < period && traj_time_dec > 0.0) {                                           // 加速接近结束且需减速
        traj_time = traj_time_dec;
        traj_accel =  copysign(d_max * a_factor, -v0);

    }
    else {
        traj_time = traj_time_acc + traj_time_flat / 1.1 + traj_time_dec / 1.1;
        traj_accel =  copysign(a_max * a_factor, -x0);
    }
}

__device__ double predictedTime(double x0, double y0, double x1, double y1, double vx, double vy) {
    double timeX, timeXAcc, timeXDec, timeXFlat, acc;
    double timeY, timeYAcc, timeYDec, timeYFlat;
    double x = x0 - x1;
    double y = y0 - y1;
    double newVelAngle = atan2(vy, vx) - atan2(y, x);
    double length = sqrt(vx * vx + vy * vy);
    x = sqrt(x * x + y * y);
    y = 0.0;
    vx = length * cospi(newVelAngle);
    vy = length * sinpi(newVelAngle);
    compute_motion_1d(x, vx, 0, 450, 450, 300, 1.5, acc, timeX, timeXAcc, timeXDec, timeXFlat);
    compute_motion_1d(y, vy, 0, 450, 450, 300, 1.5, acc, timeY, timeYAcc, timeYDec, timeYFlat);
    if(timeX < 1e-5 || timeX > 50) timeX = 0;
    if(timeY < 1e-5 || timeY > 50) timeY = 0;
    return (timeX > timeY ? timeX : timeY);
}

__device__ double predictedTheirTime(double x0, double y0, double x1, double y1, double vx, double vy) {
    double timeX, timeXAcc, timeXDec, timeXFlat, acc;
    double timeY, timeYAcc, timeYDec, timeYFlat;
    double x = x0 - x1;
    double y = y0 - y1;
    double newVelAngle = atan2(vy, vx) - atan2(y, x);
    double length = sqrt(vx * vx + vy * vy);
    x = sqrt(x * x + y * y);
    y = 0.0;
    vx = length * cospi(newVelAngle);
    vy = length * sinpi(newVelAngle);
    compute_motion_1d(x, vx, 0, 500, 500, 350, 1.5, acc, timeX, timeXAcc, timeXDec, timeXFlat);
    compute_motion_1d(y, vy, 0, 500, 500, 350, 1.5, acc, timeY, timeYAcc, timeYDec, timeYFlat);
    if(timeX < 1e-5 || timeX > 50) timeX = 0;
    if(timeY < 1e-5 || timeY > 50) timeY = 0;
    return (timeX > timeY ? timeX : timeY);
}

