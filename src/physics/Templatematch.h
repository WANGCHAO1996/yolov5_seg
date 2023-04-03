#include<iostream>
#include <vector>
#include"opencv2/core/core.hpp"
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<vector<int>> TemplateMatch(Mat img,int m);
//二维Mat元素求和函数
int mat_sum(Mat matrix);
//vector 去重复元素
vector<vector<int>>DuplicateRemoval(vector<vector<int>>res);

//打印 vector
void print_vec2(vector<vector<int>>& vec);