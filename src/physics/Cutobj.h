#include<iostream>
#include <vector>
#include"opencv2/core/core.hpp"
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Mat CutObj(vector<vector<int>>box,Mat img,int key);