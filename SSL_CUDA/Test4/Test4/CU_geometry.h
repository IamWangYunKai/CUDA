#ifndef _CU_GEOMETRY_H_
#define _CU_GEOMETRY_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "kernel.h"  
//#include <iostream>
//#include <iomanip>
//#include <cmath>
//#include <algorithm>

/*__device__ double min(double a, double b) {
	return a < b ? a : b;
}*/

class CVector {
public:
	__device__ CVector() : _x(0), _y(0) {}
	__device__ CVector(double x, double y) : _x(x), _y(y) {}
	__device__ CVector(const CVector& v) : _x(v.x()), _y(v.y()) {}
	__device__ double mod() const { return sqrt(_x * _x + _y * _y); }
	__device__ double mod2() const { return (_x * _x + _y * _y); }
	__device__ double dir() const { return atan2(y(), x()); }
	__device__ CVector rotate(double angle) const {
		double nowdir = dir() + angle;
		double nowx = mod() * cos(nowdir);
		double nowy = mod() * sin(nowdir);
		return CVector(nowx, nowy);
	}
	__device__ double x() const { return _x; }
	__device__ double y() const { return _y; }
	__device__ double value(double angle) const { return mod() * cos(dir() - angle); }
	__device__ CVector operator +(const CVector& v) const { return CVector(_x + v.x(), _y + v.y()); }
	__device__ CVector operator -(const CVector& v) const { return CVector(_x - v.x(), _y - v.y()); }
	__device__ CVector operator *(double a) const { return CVector(_x * a, _y * a); }
	__device__ double operator *(CVector b) const { return double(_x*b.x() + _y*b.y()); } //向量点乘
	__device__ CVector operator /(double a) const { return CVector(_x / a, _y / a); }
	__device__ CVector operator -() const { return CVector(-1 * _x, -1 * _y); }
	//__device__ friend std::ostream& operator <<(std::ostream& os, const CVector& v) {
	//return os << "(" << v.x() << ":" << v.y() << ")";
	//}

private:
	double _x, _y;
};

class CGeoPoint {
public:
	__device__ CGeoPoint() : _x(0), _y(0) {}
	__device__ ~CGeoPoint() {}
	__device__ CGeoPoint(double x, double y) : _x(x), _y(y) {}
	__device__ CGeoPoint(const CGeoPoint& p) : _x(p.x()), _y(p.y()) {}
	__device__ bool operator==(const CGeoPoint& rhs) { return ((this->x() == rhs.x()) && (this->y() == rhs.y())); }
	__device__ double x() const { return _x; }
	__device__ double y() const { return _y; }
	__device__ void setX(double x) { _x = x; }   // 2014/2/28 新增 设置x坐标 yys
	__device__ void setY(double y) { _y = y; }   // 2014/2/28 新增 设置y坐标 yys
	__device__ double dist(const CGeoPoint& p) const { return CVector(p - CGeoPoint(_x, _y)).mod(); }
	__device__ double dist2(const CGeoPoint& p) const { return CVector(p - CGeoPoint(_x, _y)).mod2(); }
	__device__ CGeoPoint operator+(const CVector& v) const { return CGeoPoint(_x + v.x(), _y + v.y()); }
	__device__ CGeoPoint operator*(const double& a) const { return CGeoPoint(_x * a, _y * a); }
	__device__ CVector operator-(const CGeoPoint& p) const { return CVector(_x - p.x(), _y - p.y()); }
	__device__ CGeoPoint midPoint(const CGeoPoint& p) const { return CGeoPoint((_x + p.x()) / 2, (_y + p.y()) / 2); }
	//__device__ friend std::ostream& operator <<(std::ostream& os, const CGeoPoint& v) {
	//	return os << "(" << v.x() << ":" << v.y() << ")";
	//}

private:
	double _x, _y;
};

class CGeoLine {
public:
	__device__ CGeoLine() {}
	__device__ CGeoLine(const CGeoPoint& p1, const CGeoPoint& p2) : _p1(p1), _p2(p2) {calABC();}
	__device__ CGeoLine(const CGeoPoint& p, double angle) : _p1(p), _p2(p.x() + cos(angle), p.y() + sin(angle)) {calABC();}
	__device__ void calABC() {
		if (_p1.y() == _p2.y()) {
			_a = 0;
			_b = 1;
			_c = -1.0 * _p1.y();
		}
		else {
			_a = 1;
			_b = -1.0 * (_p1.x() - _p2.x()) / (_p1.y() - _p2.y());
			_c = (_p1.x()*_p2.y() - _p1.y()*_p2.x()) / (_p1.y() - _p2.y());
		}
	}

	//投影点
	__device__ CGeoPoint projection(const CGeoPoint& p) const {
		if (_p2.x() == _p1.x()) {
			return CGeoPoint(_p1.x(), p.y());
		}
		else {
			double k = (_p2.y() - _p1.y()) / (_p2.x() - _p1.x());
			double x = (k * k * _p1.x() + k * (p.y() - _p1.y()) + p.x()) / (k * k + 1);
			double y = k * (x - _p1.x()) + _p1.y();
			return CGeoPoint(x, y);
		}
	}
	__device__ CGeoPoint point1() const { return _p1; }
	__device__ CGeoPoint point2() const { return _p2; }
	__device__ bool operator==(const CGeoLine& rhs)
	{
		return ((this->point1().x() == rhs.point1().x()) && (this->point1().y() == rhs.point1().y())
			&& (this->point2().x() == rhs.point2().x()) && (this->point2().y() == rhs.point2().y()));
	}
	__device__ const double& a() const { return _a; }
	__device__ const double& b() const { return _b; }
	__device__ const double& c() const { return _c; }
private:
	CGeoPoint _p1;
	CGeoPoint _p2;
	double _a;
	double _b;
	double _c;
};

class CGeoLineLineIntersection {
public:
	__device__ CGeoLineLineIntersection(const CGeoLine& line_1, const CGeoLine& line_2);
	__device__ bool Intersectant() const { return _intersectant; }
	__device__ const CGeoPoint& IntersectPoint() const { return _point; }
private:
	bool _intersectant;
	CGeoPoint _point;
};

class CGeoSegment : public CGeoLine {
public:
	__device__ CGeoSegment() {}
	__device__ CGeoSegment(const CGeoPoint& p1, const CGeoPoint& p2) : CGeoLine(p1, p2), _start(p1), _end(p2) {
		_compareX = abs(p1.x() - p2.x()) > abs(p1.y() - p2.y());
		_center = CGeoPoint((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2);
	}
	__device__ bool IsPointOnLineOnSegment(const CGeoPoint& p) const // 直线上的点是否在线段上
	{
		if (_compareX) {
			return p.x() > min(_start.x(), _end.x()) && p.x() < max(_start.x(), _end.x());
		}
		return p.y() > min(_start.y(), _end.y()) && p.y() < max(_start.y(), _end.y());
	}
	__device__ bool IsSegmentsIntersect(const CGeoSegment& p) const
	{
		CVector AC((start().x() - p.start().x()), (start().y() - p.start().y()));
		CVector AD((start().x() - p.end().x()), (start().y() - p.end().y()));
		CVector BC((end().x() - p.start().x()), (end().y() - p.start().y()));
		CVector BD((end().x() - p.end().x()), (end().y() - p.end().y()));
		return (((AC * AD) * (BC * BD) <= 0) && ((AC * BC) * (AD * BD) <= 0));
	}
	__device__ const CGeoPoint& start() const { return _start; }
	__device__ const CGeoPoint& end() const { return _end; }
	__device__ const CGeoPoint& center() { return _center; }

private:
	CGeoPoint _start;
	CGeoPoint _end;
	CGeoPoint _center;
	bool _compareX;
};

class CGeoShape {
public:
	__device__ virtual ~CGeoShape() { }
	__device__ virtual bool HasPoint(const CGeoPoint& p) const = 0;
};

class CGeoRectangle : public CGeoShape {
public:
	__device__ CGeoRectangle(const CGeoPoint& leftTop, const CGeoPoint& rightDown) {
		calPoint(leftTop.x(), leftTop.y(), rightDown.x(), rightDown.y());
	}
	__device__ CGeoRectangle(double x1, double y1, double x2, double y2) { calPoint(x1, y1, x2, y2); }
	__device__ void calPoint(double x1, double y1, double x2, double y2) {
		_point[0] = CGeoPoint(x1, y1);
		_point[1] = CGeoPoint(x1, y2);
		_point[2] = CGeoPoint(x2, y2);
		_point[3] = CGeoPoint(x2, y1);
	}
	__device__ virtual bool HasPoint(const CGeoPoint& p) const;
	CGeoPoint _point[4];
};

class CGeoLineRectangleIntersection {
public:
	__device__ CGeoLineRectangleIntersection(const CGeoLine& line, const CGeoRectangle& rect);
	__device__ bool intersectant() const { return _intersectant; }
	__device__ const CGeoPoint& point1() const { return _point[0]; }
	__device__ const CGeoPoint& point2() const { return _point[1]; }
private:
	bool _intersectant;
	CGeoPoint _point[2];
};

class CGeoCirlce : public CGeoShape {
public:
	__device__ CGeoCirlce() { }
	__device__ CGeoCirlce(const CGeoPoint& c, double r) : _center(c), _radius(r) { }
	__device__ virtual bool HasPoint(const CGeoPoint& p) const;
	__device__ CGeoPoint Center() const { return _center; }
	__device__ double Radius() const { return _radius; }
	__device__ double Radius2() const { return _radius*_radius; }
private:
	double _radius;
	CGeoPoint _center;
};

class CGeoLineCircleIntersection {
public:
	__device__ CGeoLineCircleIntersection(const CGeoLine& line, const CGeoCirlce& circle);
	__device__ bool intersectant() const { return _intersectant; }
	__device__ const CGeoPoint& point1() const { return _point1; }
	__device__ const CGeoPoint& point2() const { return _point2; }
private:
	bool _intersectant;
	CGeoPoint _point1;
	CGeoPoint _point2;
};

class CGeoEllipse :CGeoShape {
public:
	__device__ CGeoEllipse() { }
	__device__ CGeoEllipse(CGeoPoint c, double m, double n) : _center(c), _xaxis(m), _yaxis(n) { }
	__device__ CGeoPoint Center() const { return _center; }
	__device__ virtual bool HasPoint(const CGeoPoint& p) const;
	__device__ double Xaxis()const { return _xaxis; }
	__device__ double Yaxis()const { return _yaxis; }
private:
	double _xaxis;
	double _yaxis;
	CGeoPoint _center;
};


class CGeoLineEllipseIntersection {
public:
	__device__ CGeoLineEllipseIntersection(const CGeoLine& line, const CGeoEllipse& circle);
	__device__ bool intersectant() const { return _intersectant; }
	__device__ const CGeoPoint& point1() const { return _point1; }
	__device__ const CGeoPoint& point2() const { return _point2; }
private:
	bool _intersectant;
	CGeoPoint _point1;
	CGeoPoint _point2;
};


class CGeoSegmentCircleIntersection
{
public:
	__device__ CGeoSegmentCircleIntersection(const CGeoSegment& line, const CGeoCirlce& circle);
	__device__ bool intersectant() const { return _intersectant; }
	__device__ const CGeoPoint& point1() const { return _point1; }
	__device__ const CGeoPoint& point2() const { return _point2; }
	__device__ int size() { return intersection_size; }
private:
	bool _intersectant;
	int intersection_size;
	CGeoPoint _point1;
	CGeoPoint _point2;
};


__device__ CGeoLineLineIntersection::CGeoLineLineIntersection(const CGeoLine& line_1, const CGeoLine& line_2)
{
	double d = line_1.a() * line_2.b() - line_1.b() * line_2.a();
	if (abs(d) < 0.0001) {
		_intersectant = false;
	}
	else {
		double px = (line_1.b() * line_2.c() - line_1.c() * line_2.b()) / d;
		double py = (line_1.c() * line_2.a() - line_1.a() * line_2.c()) / d;
		_point = CGeoPoint(px, py);
		_intersectant = true;
	}
}

__device__ bool CGeoRectangle::HasPoint(const CGeoPoint& p) const
{
	int px = 0;
	int py = 0;
	for (int i = 0; i < 4; i++) { // 寻找一个肯定在多边形polygon内的点p：多边形顶点平均值
		px += _point[i].x();
		py += _point[i].y();
	}
	px /= 4;
	py /= 4;
	CGeoPoint inP = CGeoPoint(px, py);
	for (int i = 0; i < 4; i++) {
		CGeoSegment line_1(p, inP);
		CGeoSegment line_2(_point[i], _point[(i + 1) % 4]);
		CGeoLineLineIntersection inter(line_1, line_2);
		if (inter.Intersectant()
			&& line_1.IsPointOnLineOnSegment(inter.IntersectPoint())
			&& line_2.IsPointOnLineOnSegment(inter.IntersectPoint())) {
			return false;
		}
	}
	return true;
}

__device__ bool CGeoCirlce::HasPoint(const CGeoPoint& p) const
{
	double d = (p - _center).mod();
	if (d < _radius) {
		return true;
	}
	return false;
}

__device__ bool CGeoEllipse::HasPoint(const CGeoPoint& p) const
{
	double x = p.x() - _center.x();
	double y = p.y() - _center.y();
	if ((x*x / (_xaxis*_xaxis) + y*y / (_yaxis*_yaxis)) < 1) {
		return true;
	}
	return false;
}


__device__ CGeoLineRectangleIntersection::CGeoLineRectangleIntersection(const CGeoLine& line, const CGeoRectangle& rect)
{
	int num = 0;
	for (int i = 0; i < 4; i++) {
		CGeoSegment rectbound(rect._point[i], rect._point[(i + 1) % 4]);
		CGeoLineLineIntersection inter(line, rectbound);
		if (inter.Intersectant() && rectbound.IsPointOnLineOnSegment(inter.IntersectPoint())) {
			if (num >= 2) {
				printf("Error in CGeoLineRectangleIntersection, Num be: %d", num);
			}
			_point[num] = inter.IntersectPoint();
			num++;
		}
	}
	if (num > 0) {
		_intersectant = true;
	}
	else {
		_intersectant = false;
	}
}

__device__ CGeoLineCircleIntersection::CGeoLineCircleIntersection(const CGeoLine& line, const CGeoCirlce& circle)
{
	CGeoPoint projection = line.projection(circle.Center());
	CVector center_to_projection = projection - circle.Center();
	double projection_dist = center_to_projection.mod();
	if (projection_dist < circle.Radius()) { // 圆心到直线的距离小于半径
		_intersectant = true;
		if (projection_dist < 0.001) {
			CVector p1top2 = line.point2() - line.point1();
			_point1 = circle.Center() + p1top2 * circle.Radius() / (p1top2.mod() + 0.0001);
			_point2 = circle.Center() + p1top2 * circle.Radius() / (p1top2.mod() + 0.0001) * -1;
		}
		else {
			double angle = acos(projection_dist / circle.Radius());
			_point1 = circle.Center() + center_to_projection.rotate(angle) * circle.Radius() / projection_dist;
			_point2 = circle.Center() + center_to_projection.rotate(-angle) * circle.Radius() / projection_dist;
		}
	}
	else {
		_intersectant = false;
	}
}

__device__ CGeoLineEllipseIntersection::CGeoLineEllipseIntersection(const CGeoLine& line, const CGeoEllipse& ellipse)
{
	const CGeoPoint Center = ellipse.Center();
	double x0 = Center.x();
	double y0 = Center.y();
	double m = ellipse.Xaxis();
	double n = ellipse.Yaxis();
	double a = line.a();
	double b = line.b();
	double c = line.c() + a*x0 + b*y0;
	double x1 = 0;
	double y1 = 0;
	double x2 = 0;
	double y2 = 0;
	//先求出与标准方程的交点再进行平移
	if (b != 0) {
		double p = (n*n*b*b + m*m*a*a);
		double q = (2 * m*m*a*c);
		double r = (-m*m*n*n*b*b + (m*m*c*c));
		if ((q*q - 4 * p*r)>0) {
			_intersectant = true;
			x1 = ((-1)*q + sqrt(q*q - 4 * p*r)) / (2 * p);
			y1 = -(a*x1 + c) / b;
			x2 = ((-1)*q - sqrt(q*q - 4 * p*r)) / (2 * p);
			y2 = -(a*x2 + c) / b;
			CVector transVector1 = CVector(x1, y1);
			CVector transVector2 = CVector(x2, y2);
			_point1 = Center + transVector1;
			_point2 = Center + transVector2;
		}
		else if ((q*q - 4 * p*r) == 0) {
			_intersectant = true;
			x1 = (-q) / (2 * p);
			y1 = -(a*x1 + c) / b;
			CVector transVector = CVector(x1, y1);
			_point1 = Center + transVector;
			_point2 = Center + transVector;
		}
		else {
			_intersectant = false;
		}
	}
	else if (b == 0) {
		double t = m*m*n*n - c*c*n*n / (a*a);
		if (t > 0) {
			_intersectant = true;
			x1 = (-c) / a;
			y1 = sqrt(t) / m;
			x2 = (-c) / a;
			y2 = -sqrt(t) / m;
			CVector transVector1 = CVector(x1, y1);
			CVector transVector2 = CVector(x2, y2);
			_point1 = Center + transVector1;
			_point2 = Center + transVector2;
		}
		else if (t == 0) {
			_intersectant = true;
			x1 = (-c) / a;
			y1 = 0;
			CVector transVector = CVector(x1, y1);
			_point1 = Center + transVector;
			_point2 = Center + transVector;
		}
		else {
			_intersectant = false;
		}
	}
}

__device__ CGeoSegmentCircleIntersection::CGeoSegmentCircleIntersection(const CGeoSegment& line, const CGeoCirlce& circle)
{
	CGeoPoint projection = line.projection(circle.Center());
	CVector center_to_projection = projection - circle.Center();
	double projection_dist = center_to_projection.mod();
	if (projection_dist < circle.Radius()) { // 圆心到直线的距离小于半径
		_intersectant = true;intersection_size = 2;
		if (projection_dist < 0.001) {
			CVector p1top2 = line.point2() - line.point1();
			_point1 = circle.Center() + p1top2 * circle.Radius() / (p1top2.mod() + 0.0001);
			_point2 = circle.Center() + p1top2 * circle.Radius() / (p1top2.mod() + 0.0001) * -1;
		}
		else {
			double angle = acos(projection_dist / circle.Radius());
			_point1 = circle.Center() + center_to_projection.rotate(angle) * circle.Radius() / projection_dist;
			_point2 = circle.Center() + center_to_projection.rotate(-angle) * circle.Radius() / projection_dist;
		}
		if (!line.IsPointOnLineOnSegment(_point2) && (!line.IsPointOnLineOnSegment(_point1))) { _intersectant = false;intersection_size = 0; }
		else {
			if (!line.IsPointOnLineOnSegment(_point1)) { _point1 = _point2;intersection_size = 1; }
			if (!line.IsPointOnLineOnSegment(_point2)) { intersection_size = 1; }
		}
	}
	else {
		_intersectant = false;intersection_size = 0;
	}
}


#endif