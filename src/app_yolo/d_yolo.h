#include<iostream>
#include<opencv2/opencv.hpp>


struct OutputSeg{
    int id;             //结果类别id
    float confidence;   //结果置信度
    cv::Rect box;       //矩形框  //x,y,w,h
    cv::Mat boxMask;    //矩形框内的mask
};