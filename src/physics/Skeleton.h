#include<iostream>
#include <vector>
#include<math.h>
#include"opencv2/core/core.hpp"
#include<opencv2/opencv.hpp>
#include<opencv2/ximgproc.hpp>
using namespace std;
using namespace cv;
//提取中线函数声明
Mat Thin_Skeleton(Mat image);
void Clear_MicroConnected_Areas(cv::Mat src, cv::Mat &dst, double min_area);