#include<iostream>
#include <vector>
#include"opencv2/core/core.hpp"
#include<opencv2/opencv.hpp>
#include<algorithm>

using namespace std;
using namespace cv;

float DisWireSlip(vector<vector<int>>a,vector<vector<int>>b); //wire  slip

float edudis(vector<int>wire,vector<int>slip); 

vector<int> SlipPosLocal(vector<vector<int>>aa,vector<vector<int>>bb,float pos_temp); // wire slip_pos pos_min

float MaxSlider(vector<vector<int>>slider,vector<int>pos); //slider  slip_pos
