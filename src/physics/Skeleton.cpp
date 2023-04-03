#include"Skeleton.h"




/**
* @brief  Clear_MicroConnected_Areas         清除微小面积连通区函数
* @param  src                                输入图像矩阵
* @param  dst                                输出结果
* @return min_area                           设定的最小面积清除阈值
*/
void Clear_MicroConnected_Areas(cv::Mat src, cv::Mat &dst, double min_area)
{
    // 备份复制
	dst = src.clone();
	std::vector<std::vector<cv::Point> > contours;  // 创建轮廓容器
	std::vector<cv::Vec4i> 	hierarchy;  
 
	// 寻找轮廓的函数
	// 第四个参数CV_RETR_EXTERNAL，表示寻找最外围轮廓
	// 第五个参数CV_CHAIN_APPROX_NONE，表示保存物体边界上所有连续的轮廓点到contours向量内
	cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
 
	if (!contours.empty() && !hierarchy.empty()) 
	{
		std::vector<std::vector<cv::Point> >::const_iterator itc = contours.begin();
		// 遍历所有轮廓
		while (itc != contours.end()) 
		{
			// 定位当前轮廓所在位置
			cv::Rect rect = cv::boundingRect(cv::Mat(*itc));
			// contourArea函数计算连通区面积
			double area = contourArea(*itc);
			// 若面积小于设置的阈值
			if (area < min_area) 
			{
				// 遍历轮廓所在位置所有像素点
				for (int i = rect.y; i < rect.y + rect.height; i++) 
				{
					uchar *output_data = dst.ptr<uchar>(i);
					for (int j = rect.x; j < rect.x + rect.width; j++) 
					{
						// 将连通区的值置0
						if (output_data[j] == 255) 
						{
							output_data[j] = 0;
						}
					}
				}
			}
			itc++;
		}
	}
}


//定义Skeleton函数
Mat Thin_Skeleton(Mat image)
{
    Mat thin_img;
    threshold(image,image,125,255,THRESH_BINARY);//二值化
    image=255-image;//取反

    //形态学处理应该可以简化
    Mat element;
    element = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(image,image,element,Point(-1,-1),3);
    medianBlur(image,image,5);
    blur(image,image,Point(3,3));

    //过滤短线
    Clear_MicroConnected_Areas(image,image,20);//20是面积

    // //opencv 扩展包的细化算法
    ximgproc::thinning(image,thin_img);

    // //find contours 查找边
    // vector<vector<cv::Point>>contours;
    // vector<cv::Vec4i>hierarchy;

    // findContours(image,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    // //draw result
    // for(size_t i=0;i<contours.size();i++)
    // {
    //     cout<<i<<endl;
    //     Scalar color=Scalar(255);
    //     drawContours(image,contours,(int)i,color,-1,8,hierarchy,0);
    // }
    //threshold(image,image,0,255,THRESH_BINARY|THRESH_OTSU);//二值化
    return thin_img;
    


}
