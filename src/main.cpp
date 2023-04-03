// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>
//#include <cuda.hpp>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp> 


#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <app_yolo/yolo.hpp>
#include <app_http/http_server.hpp>
#include "app_cls/cls.hpp"
#include"physics/Calculatloop.h"
#include"physics/Switch.h"
#include"physics/Skeleton.h"
#include"physics/Templatematch.h"
#include"physics/Slipwire.h"
#include"physics/Power.h"
#include"physics/Lightwire.h"
#include"physics/Cutlight.h"
#include"physics/Cutobj.h"
#include<opencv2/freetype.hpp>
#include<common/cc_util.hpp>


using namespace std;;
using namespace cv;
//定义全局变量
int g_col=0;
int g_row=0;
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
// static const char* cocolabels[] = {"switch","switch-","switch+","switch head", "light","light bulb",
//                                     "light+ ","light-","power","power-","power+","slip resister",
//                                     "slip resister-","slip resister+","slider","wire "
// };
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}
static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

class LogicalController : public Controller{
	SetupController(LogicalController);

public:
	bool startup();
 
public: 
	DefRequestMapping(detect_seg);
    DefRequestMapping(getReturn);
    DefRequestMapping(detect_seg_ai);
    DefRequestMapping(putBase64Image);
    DefRequestMapping(getFile);
    DefRequestMapping(getRes);
    DefRequestMapping(getImg);
     DefRequestMapping(getBinary);


private:
    shared_ptr<Yolo::Infer> yolo_;//det+mask
    shared_ptr<EnginePool> cls_;//cls
};

Json::Value LogicalController::detect_seg_ai(const Json::Value& param)
{
    //最终的得分
    int res_score=0;
    int total_score=0;
    //开关在连接成电路时是否断开
    bool switch_flag=true;
    // //电源在首次回路中连接是否正确
    // bool power_flag=true;
    // 滑阻器在首次回路中连接是否正确
    bool slip_flag=true;
    // 电灯在首次回路中连接是否正确
    bool light_flag=true;
    //电阻器划块
    bool slider_flag=true;

    //实现写中文
    string text1="未连接成正确回路!";
    string text2="开关状态:--";
    string text3="滑阻器接线:--";
    string text4="电源接线:--";
    string text5="电灯接线:--";
    string text6="滑块阻值:--";
    cv::Ptr<cv::freetype::FreeType2>ft2;
    ft2=cv::freetype::createFreeType2();
    ft2->loadFontData("/usr/share/fonts/SimHei.ttf",0);

    //加载分类标签
    auto cls_labels=load_labels("cls.txt");


    cout<<"11...."<<endl;
    auto session = get_current_session();
    if(session->request.body.empty())
        return failure("Request body is empty");


    
    //读取视频
    VideoCapture capture(session->request.body);
    // VideoCapture capture = session->request.body;
    cout<<"1...."<<endl;
    if (!capture.isOpened()){
		return -1;
	}
	int imgIndex = 0;
    Mat frame;
	capture >> frame;
    // frame = imdecode(frame, 1);
    
    //处理帧
    while(!frame.empty())
    {   Json::Value out(Json::arrayValue);
        INFO("***读取视频处理***");
        string outputpath = "video/img/" + to_string(imgIndex) + ".jpg";
        //string maskpath = "video/mask/" + to_string(imgIndex) + ".jpg";
        Mat image = frame;
        Mat mask_seg=image.clone();
        Mat cut_img=image.clone();
        g_col=image.cols;
        g_row=image.rows;

        //建立一个空白
        cv::Mat mask_te(image.rows,image.cols,CV_8UC1,cv::Scalar(255));//创建一通道矩阵，每个像素都是255
        //创建列表 存放box信息
        vector<vector<int>> bbox_rect;
        vector<int> bbox_rect_index={0,4,8,11,15};//存放 switch, light, power,  slip resister, wire   标签索引
        //创建switch box信息
        vector<vector<int>> bbox_switch;
        vector<int> bbox_switch_index={1,2,3}; //switch-,switch+,switch head
        //创建滑阻器正负极端子bbox
        vector<vector<int>> bbox_slip;
        vector<int> bbox_slip_index={12,13};//slip resister- ,slip resister+
        //创建滑阻器的划块
        vector<vector<int>> bbox_slider;
        vector<int> bbox_slider_index={14};
        //创建滑阻器的正极
        vector<vector<int>> bbox_slip_pos;
        vector<int> bbox_slip_pos_index={13};
        //创建电源
        vector<vector<int>> bbox_power;
        vector<int> bbox_power_index={9,10}; //power- , power+
        //创建电灯
        vector<vector<int>> bbox_light;
        vector<int> bbox_light_index={6,7}; //light+ , light-

        //mask
        auto boxes = yolo_->commit(image).get();
       
        for(auto& box : boxes)
        {
            //颜色
            uint8_t r,g,b;
            tie(r,g,b) = iLogger::random_color(box.class_label+86);//这个地方随意哈 为了区分颜色
            //绘制box
            cv::rectangle(image,cv::Point(box.left,box.top),cv::Point(box.right,box.bottom),cv::Scalar(b,g,r),1);

            int label = static_cast<int>(box.class_label);
            auto name    = cocolabels[label];
            auto caption = iLogger::format("%s %.2f", name, box.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 1, nullptr).width + 10;
            cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + width, box.top), cv::Scalar(b, g, r), -1);//绘制label的框
            cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
            
            //绘制mask 
            mask_seg(cv::Rect(box.left,box.top,box.right-box.left,box.bottom-box.top)).setTo(cv::Scalar(b, g, r), box.mask);

            //mask wire only
            if(box.class_label==15) //15 导线label索引
            {
                mask_te(cv::Rect(box.left,box.top,box.right-box.left,box.bottom-box.top)).setTo(cv::Scalar(b, g, r), box.mask);
            }

            //拿到需要的判断数据
            vector<int>bbox_rect_temp;      //元器件框
            vector<int>bbox_switch_temp;    //开关
            vector<int>bbox_slip_temp;      //滑阻器
            vector<int>bbox_slip_pos_temp;  //滑阻器正极
            vector<int>bbox_slider_temp;    //划块
            vector<int>bbox_power_temp;     //电源
            vector<int>bbox_light_temp;     //电灯

            //将box信息写入bbox_rect  电路回路
            bool present  = std::find(begin(bbox_rect_index),end(bbox_rect_index),box.class_label) != end(bbox_rect_index); //判断元器件lable 
            bool present2 = std::find(begin(bbox_switch_index),end(bbox_switch_index),box.class_label) != end(bbox_switch_index); //switch label
            bool present3 = std::find(begin(bbox_slip_index),end(bbox_slip_index),box.class_label) != end(bbox_slip_index); //slip label
            bool present4 = std::find(begin(bbox_slip_pos_index),end(bbox_slip_pos_index),box.class_label) != end(bbox_slip_pos_index); //slip+ label
            bool present5 = std::find(begin(bbox_slider_index),end(bbox_slider_index),box.class_label) != end(bbox_slider_index); //slider label
            bool present6 = std::find(begin(bbox_power_index),end(bbox_power_index),box.class_label) != end(bbox_power_index);
            bool present7 = std::find(begin(bbox_light_index),end(bbox_light_index),box.class_label) != end(bbox_light_index);
            if(present)
            {
                bbox_rect_temp.push_back(box.class_label);
                bbox_rect_temp.push_back(box.left);
                bbox_rect_temp.push_back(box.top);
                bbox_rect_temp.push_back(box.right);
                bbox_rect_temp.push_back(box.bottom);
                bbox_rect.push_back(bbox_rect_temp);
            }
            if(present2)
            {
                bbox_switch_temp.push_back(box.class_label);
                bbox_switch_temp.push_back(box.left);
                bbox_switch_temp.push_back(box.top);
                bbox_switch_temp.push_back(box.right);
                bbox_switch_temp.push_back(box.bottom);
                bbox_switch.push_back(bbox_switch_temp);
            }
            if(present3)
            {
                bbox_slip_temp.push_back(box.class_label);
                bbox_slip_temp.push_back(box.left/2+box.right/2);
                bbox_slip_temp.push_back(box.top/2+box.bottom/2);
                bbox_slip.push_back(bbox_slip_temp);
            }
            if(present4)
            {
                bbox_slip_pos_temp.push_back(box.class_label);
                bbox_slip_pos_temp.push_back(box.left/2+box.right/2);
                bbox_slip_pos_temp.push_back(box.top/2+box.bottom/2);
                bbox_slip_pos.push_back(bbox_slip_pos_temp);
            }
            if(present5)
            {
                bbox_slider_temp.push_back(box.class_label);
                bbox_slider_temp.push_back(box.left/2+box.right/2);
                bbox_slider_temp.push_back(box.top/2+box.bottom/2);
                bbox_slider.push_back(bbox_slider_temp);
            }
            if(present6)
            {
                bbox_power_temp.push_back(box.class_label);
                bbox_power_temp.push_back(box.left/2+box.right/2);
                bbox_power_temp.push_back(box.top/2+box.bottom/2);
                bbox_power.push_back(bbox_power_temp);
            }
            if(present7)
            {
                bbox_light_temp.push_back(box.class_label);
                bbox_light_temp.push_back(box.left/2+box.right/2);
                bbox_light_temp.push_back(box.top/2+box.bottom/2);
                bbox_light.push_back(bbox_light_temp);
            }
        }
        addWeighted(image, 0.6, mask_seg, 0.5, 0, image); //将mask加在原图上面 
        //imwrite("out.jpg",image);//mask+det

        //处理wire
        mask_te=Thin_Skeleton(mask_te);
        vector<vector<int>> wire_res;

        //初始化
        Json::Value item;
        item["目前得分"] = 0;
        item["得分点1"] = 0;
        item["得分点2"] = 0;
        item["得分点3"] = 0;
        item["得分点4"] = 0;
        item["得分点5"] = 0;
        item["最终得分"] = "--";

        //判断电路逻辑
        int circle=CalculatingLoop(bbox_rect);
        if(circle==8 && res_score==0 && switch_flag == true && light_flag==true)
        {
            vector<float> res_switch = SwitchDistance(bbox_switch);
            bool state_switch = SwitchState(res_switch);
            printf("state_switch=%d\n",state_switch);
            if(!state_switch)
            {
                INFO("开关处于断开状态，进入滑阻器连接判断!");

                //导线wire端点  vector
                wire_res=TemplateMatch(mask_te,3);
                INFO("导线端点获取完毕，判断滑阻器连接情况！");
                
                //增加导线的端子判断 当端子个数不为指定 跳过下面判断 
                if(wire_res.size()==8)
                {
                    res_score+=1;//switch +1 分
                    item["得分点1"]=1;
                    Mat cut_img1=CutObj(bbox_rect,cut_img,0);
                    imwrite("testout/switchopen.jpg",cut_img1);

                     //计算导线端点和滑阻器端子距离，判断滑阻器连接情况  box_slip  返回wire与slip正极最小距离 
                    float slip_pos = DisWireSlip(wire_res,bbox_slip);
                    if(slip_pos != NULL)
                    {
                        INFO("滑阻器连接正常，判断滑阻器的阻值情况!");
                        ft2->putText(image,"滑阻器连接: 正确",Point(1500,150),30,Scalar(127,255,0),3,8,true);
                        vector<int> pos = SlipPosLocal(wire_res,bbox_slip_pos,slip_pos);
                        float dis = MaxSlider(bbox_slider,pos);
                        if(dis!=NULL)
                        {
                            res_score+=1; //+1分
                            item["得分点2"]=1;
                            Mat cut_img2=CutObj(bbox_rect,cut_img,3);
                            imwrite("testout/OKslip.jpg",cut_img2);
                            INFO("滑阻器阻值处于最大，判断电源连接情况！");
                            ft2->putText(image,"滑块阻值: 处于最大",Point(1500,200),30,Scalar(127,255,0),3,8,true);
                            
                        }
                        if(dis==NULL)
                        {//slider在首次回路为在最大值
                            slider_flag=false;
                            cout<<"滑阻器为在最大值"<<endl;
                            ft2->putText(image,"滑块阻值: 未处于最大",Point(1500,200),30,Scalar(127,255,0),3,8,true);
                        }
                    }
                    else{
                        //如果滑阻器的连接不正确： 两正 两负 未接上 slip_flag 置为 false  输出目前得分
                        slip_flag=false;   
                             

                    }

                    //判断电源的连接情况 正确：返回值  错误 返回NULL 得分统计+1
                    float power_res = WirePower(wire_res,bbox_power);
                    if(power_res) 
                    {
                        res_score +=1;
                        item["得分点3"]=1;
                        Mat cut_img3=CutObj(bbox_rect,cut_img,1);
                        imwrite("testout/OKpower.jpg",cut_img3);
                        ft2->putText(image,"电源连接: 正确",Point(1500,250),30,Scalar(127,255,0),3,8,true);
                    }else
                    {
                        Mat cut_img3=CutObj(bbox_rect,cut_img,1);
                        imwrite("testout/NGpower.jpg",cut_img3);
                        ft2->putText(image,"电源连接: 错误",Point(1500,250),30,Scalar(127,255,0),3,8,true);
                    }
                    

                    //判断电灯连接情况 正确：返回值  错误 返回NULL 得分统计+1
                    INFO("判断电灯泡连接情况！");

                    float light_res = WireLight(wire_res,bbox_light);
                    if(light_res) 
                    {
                        res_score+=1;
                        item["得分点4"]=1;
                        Mat cut_img4=CutObj(bbox_rect,cut_img,2);
                        imwrite("testout/OKLight.jpg",cut_img4);
                        ft2->putText(image,"电灯连接: 正确",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                    }
                    else{
                        ft2->putText(image,"电灯连接: 错误",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                        light_flag=false;
                    }
                    
                    //CutLight(bbox_rect,cut_img);

                }
                //统计得分 + 截图
                item["目前得分"]=res_score;
                String count = "目前实验得分:" + to_string(res_score);
                ft2->putText(image, count, Point(1500, 350), 30, cv::Scalar(255, 0, 255), 2, 8,true);
                
            }
            else
            {
                //开关在闭合电路前 是闭合的 则短路 总分直接为0
                switch_flag=false;
                ft2->putText(image,"实验打分完毕...", Point(1500, 50),30, cv::Scalar(255, 0, 255), 2, 8,true);
                ft2->putText(image,"开关状态: 闭合",Point(1500,100),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"电路短路,最终得分: 0",Point(100,100),30,Scalar(255,0,255),2,8,true);
                // item["最终得分"]=0;
                // out.append(item);
                // return success(out);

            }
        }
        //不是回路
        else if((circle!=8 && res_score==0) && switch_flag!=false)
        {
            //没有连接成回路,且未短路 
            ft2->putText(image,text1,Point(1500,50),30,Scalar(0,0,255),2,8,true);
            ft2->putText(image,text2,Point(1500,100),30,Scalar(127,255,0),2,8,true);
            ft2->putText(image,text3,Point(1500,150),30,Scalar(127,255,0),2,8,true);
            ft2->putText(image,text6,Point(1500,200),30,Scalar(127,255,0),2,8,true);
            ft2->putText(image,text4,Point(1500,250),30,Scalar(127,255,0),2,8,true);
            ft2->putText(image,text5,Point(1500,300),30,Scalar(127,255,0),2,8,true);
            String count = "目前实验得分:" + to_string(res_score);
		    ft2->putText(image, count, Point(1500, 350), 30, cv::Scalar(255, 0, 255), 2, 8,true);
            out.append(item);
            return success(out);
        }
        else if(switch_flag==false)
        {
            Mat cut_img1=CutObj(bbox_rect,cut_img,1);
            imwrite("testout/switchclose.jpg",cut_img1);
            ft2->putText(image, "实验打分完毕...", Point(1500, 50), 30, cv::Scalar(186, 85, 211), 2, 8,true);
            String count = "最终得分:" + to_string(res_score)+"分";
		    ft2->putText(image, count, Point(100, 100), 35, cv::Scalar(0, 128, 255), 2, 8,true);
            ft2->putText(image, "电路形成回路时,开关处于闭合状态:短路...", Point(100, 150), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            item["最终得分"]=0;
            out.append(item);
            return success(out);
        }
        else if(slip_flag==false)//滑阻器连接错误
        {
            //统计其他的得分情况 返回
            item["最终得分"]=res_score;
            out.append(item);
            return success(out);

            ft2->putText(image, "实验打分完毕...", Point(1500, 50), 30, cv::Scalar(186, 85, 211), 2, 8,true);
            String count = "最终得分:" + to_string(res_score)+"分";
		    ft2->putText(image, count, Point(100, 100), 35, cv::Scalar(0, 128, 255), 2, 8,true);
            ft2->putText(image, "电路形成回路时,开关处于断开状态:1分", Point(100, 150), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "滑阻器连接错误，一极被多次连接:0分", Point(100, 200), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "电源连接正确:1分", Point(100, 250), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "电灯连接正确:1分", Point(100, 300), 30, cv::Scalar(153, 51, 255), 2, 8,true);       
        }
        if(circle==8 && res_score==2 && switch_flag == true && light_flag==false)
        {
            //情况：首次回路中 s得witch断开 powwer连接正常  祜族其连接正常但是slider未在最大 但是light一级的导线被一个挡住导致判断未连接 了2分
            ft2->putText(image,"电路连接成正确回路！",Point(1500,50),30,Scalar(0,0,255), 3, 8, true);
            vector<float> res_switch = SwitchDistance(bbox_switch);
            bool state_switch3 = SwitchState(res_switch);
            if(!state_switch3)
            {
                INFO("开关处于断开状态，进入滑阻器连接判断!");
                
                
                //导线wire端点  vector
                wire_res=TemplateMatch(mask_te,3);
                INFO("导线端点获取完毕，判断滑阻器连接情况！");
                
                //增加导线的端子判断 当端子个数不为指定 跳过下面判断 
                if(wire_res.size()==8)
                {
                    ft2->putText(image,"开关状态: 断开",Point(1500,100),30,Scalar(127,255,0),3,8, true);
                    ft2->putText(image,"滑阻器连接: 正确",Point(1500,150),30,Scalar(127,255,0),3,8,true);
                    ft2->putText(image,"滑块阻值: 未处于最大",Point(1500,200),30,Scalar(127,255,0),3,8,true);
                    ft2->putText(image,"电源连接: 正确",Point(1500,250),30,Scalar(127,255,0),3,8,true);
                   
                    //判断电灯连接情况 正确：返回值  错误 返回NULL 得分统计+1
                    INFO("判断电灯泡连接情况！");

                    float light_res1 = WireLight(wire_res,bbox_light);
                    if(light_res1) 
                    {
                        res_score+=1;
                        item["得分点4"]=1;
                        Mat cut_img4=CutObj(bbox_rect,cut_img,2);
                        imwrite("testout/OKLight.jpg",cut_img4);
                        ft2->putText(image,"电灯连接: 正确",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                    }
                    else{
                        ft2->putText(image,"电灯连接: 错误",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                        light_flag=false;
                    }
                    
                    //CutLight(bbox_rect,cut_img);
                }
            }

            item["目前得分"]=res_score;
            out.append(item);
            return success(out);
        }
        //此时可能存在滑阻器阻值在首次回路时未在最大,闭合开关前再进行调节的情况
        if(circle==8 && res_score==3 && !slider_flag  && total_score!=3 )//此时的slider_flag=false
        {   ft2->putText(image,"电路连接成回路,滑块阻值未在最大!",Point(1500,50),30,Scalar(0,0,255),2,8,true);
            vector<float> res_switch1 = SwitchDistance(bbox_switch);
            bool state_switch1 = SwitchState(res_switch1);
            if(!state_switch1)
            {
                ft2->putText(image,"开关状态: 断开",Point(1500,100),30,Scalar(127,255,0),2,8,true);
                //导线wire端点  vector
                wire_res=TemplateMatch(mask_te,3);
                INFO("导线端点获取完毕，判断滑阻器连接情况！");

                //计算导线端点和滑阻器端子距离，判断滑阻器连接情况  box_slip  返回wire与slip正极最小距离 
                float slip_pos = DisWireSlip(wire_res,bbox_slip);
                if(slip_pos != NULL)
                {
                    ft2->putText(image,"滑阻器连接:正确",Point(1500,150),30,Scalar(127,255,0),2,8,true);
                    vector<int> pos = SlipPosLocal(wire_res,bbox_slip_pos,slip_pos);
                    float dis = MaxSlider(bbox_slider,pos);
                    if(dis!=NULL)
                    {
                        res_score+=1; //+1分
                        item["得分点2"]=1;
                        Mat cut_img2=CutObj(bbox_rect,cut_img,3);
                        imwrite("testout/OKslip.jpg",cut_img2);
                        INFO("滑阻器阻值被调节到最大！");
                        ft2->putText(image,"滑块阻值: 处于最大",Point(1500,200),30,Scalar(127,255,0),2,8,true);
                    }
                    else
                    {//slider在首次回路为在最大值
                        slider_flag=false;
                        ft2->putText(image,"滑块阻值: 未处于最大",Point(1500,200),30,Scalar(127,255,0),2,8,true);
                    }
                }
                ft2->putText(image,"电源连接: 正确",Point(1500,250),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"电灯连接: 正确",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                String count = "目前实验得分:" + to_string(res_score);
		        ft2->putText(image, count, Point(1500, 350), 30, cv::Scalar(255, 0, 255), 2, 8,true);
                item["目前得分"]=res_score;
                out.append(item);
                return success(out);
            }
            //得分==3 滑阻器没在最大 直接闭合电路 得分最终在三分
            if(state_switch1)
            {
                ft2->putText(image,"开关状态: 闭合",Point(1500,100),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image, "实验打分完毕...", Point(1500, 150), 30, cv::Scalar(186, 85, 211), 2, 8,true);
                slider_flag = true;
                   
                String count = "最终得分:" + to_string(res_score)+"分";
                ft2->putText(image, count, Point(100, 100), 35, cv::Scalar(0, 128, 255), 2, 8,true);
                ft2->putText(image, "电路形成回路时,开关处于断开状态:1分", Point(100, 150), 30, cv::Scalar(153, 51, 255), 2, 8,true);
                ft2->putText(image, "滑阻器连接正确，闭合电路前滑块未处于最大阻值:0分", Point(100, 200), 30, cv::Scalar(153, 51, 255), 2, 8,true);
                ft2->putText(image, "电源连接正确:1分", Point(100, 250), 30, cv::Scalar(153, 51, 255), 2, 8,true);
                ft2->putText(image, "电灯连接正确:1分", Point(100, 300), 30, cv::Scalar(153, 51, 255), 2, 8,true);
                total_score=res_score;
                Mat cut_img2=CutObj(bbox_rect,cut_img,3);
                imwrite("testout/NGslip.jpg",cut_img2); 
            }

        }

        //第二阶段 再次出现回路
        //判断switch状态
        //当电路有回路且开关闭合
        if(res_score==4)
        {
            ft2->putText(image,"电路连接成回路,判断灯泡的亮度变化!",Point(1400,50),30,Scalar(0,0,255),2,8,true);
            //再次判断switch
            vector<float> res_switch2 = SwitchDistance(bbox_switch);
            bool state_switch2 = SwitchState(res_switch2);
            if(!state_switch2)
            {
                ft2->putText(image,"开关状态:断开 ",Point(1500,100),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"滑阻器连接:正确",Point(1500,150),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"滑块阻值:处于阻值最大",Point(1500,200),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"电源连接:正确",Point(1500,250),30,Scalar(127,255,0),2,8,true);
                item["目前得分"]=res_score;
                out.append(item);
                return success(out);
            }
            if(state_switch2)
            {
                ft2->putText(image,"开关状态:闭合 ",Point(1500,100),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"滑阻器连接:正确",Point(1500,150),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"滑块阻值:处于阻值最大",Point(1500,200),30,Scalar(127,255,0),2,8,true);
                ft2->putText(image,"电源连接:正确",Point(1500,250),30,Scalar(127,255,0),2,8,true);
                
                INFO("开关闭合，判断电灯亮灭情况");
                //取出light图像 定义函数 返回图像
                cut_img=CutLight(bbox_rect,cut_img);
                INFO("截取Light图像成功");
                //最好是push到一个队列里面 然后分类模型在里面取
                //判断灯泡亮不亮
                //二分类模型
                //imwrite("cutimg1.jpg",cut_img);
                auto cls_light_fut=cls_->commit(cut_img);
                //INFO("one.....");
                auto cls_res=cls_light_fut.get();
                putText(cut_img,cls_labels[cls_res.label].c_str(),Point(25,25),FONT_HERSHEY_SIMPLEX,1,Scalar(0,234,255),2,8,false);
                printf("Predict: %d,%f,%s\n", cls_res.confidence,cls_res.label,cls_labels[cls_res.label].c_str());
                cvtColor(cut_img,cut_img,cv::COLOR_RGB2BGR);
                // imwrite("testout/switch.jpg",cut_img);

                if(cls_res.label==0) 
                {
                    res_score+=2;
                    item["得分点5"]=2;
                    ft2->putText(image,"灯泡状态: 亮",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                    imwrite("testout/lightbright.jpg",cut_img);
                }
                else{

                    ft2->putText(image,"灯泡状态: 不亮",Point(1500,300),30,Scalar(127,255,0),2,8,true);
                }
                        
            }
            String count = "目前实验得分:" + to_string(res_score);
		    ft2->putText(image, count, Point(1500, 350), 30, cv::Scalar(255, 0, 255), 2, 8,true);
            item["目前得分"]=res_score;
            out.append(item);
            return success(out);

        }
        if(res_score==6)
        {
            ft2->putText(image, "实验打分完毕...", Point(1500, 50), 30, cv::Scalar(186, 85, 211), 2, 8,true);
            String count = "最终得分:" + to_string(res_score)+"分";
		    ft2->putText(image, count, Point(100, 100), 35, cv::Scalar(0, 128, 255), 2, 8,true);
            ft2->putText(image, "电路形成回路时,开关处于断开状态:1分", Point(100, 150), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "滑阻器连接正确，且闭合电路前滑块处于最大阻值:1分", Point(100, 200), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "电源连接正确:1分", Point(100, 250), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "电灯连接正确:1分", Point(100, 300), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "闭合开关后,调节滑块灯泡亮度变化:2分", Point(100, 350), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            item["目前得分"]=res_score;
            item["最终得分"]=resetiosflags;
            out.append(item);
            return success(out);
        }
        
        cout<<"目前得分:"<<res_score<<endl;
       
        
        cout<<"total:"<<total_score<<endl;
        //输出 文字
        if(total_score==3)
        {
            ft2->putText(image, "实验打分完毕...", Point(1500, 50), 30, cv::Scalar(186, 85, 211), 2, 8,true);
    
            String count = "最终得分:" + to_string(total_score)+"分";
            ft2->putText(image, count, Point(100, 100), 35, cv::Scalar(0, 128, 255), 2, 8,true);
            ft2->putText(image, "电路形成回路时,开关处于断开状态:1分", Point(100, 150), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "滑阻器连接正确，闭合电路前滑块未处于最大阻值:0分", Point(100, 200), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "电源连接正确:1分", Point(100, 250), 30, cv::Scalar(153, 51, 255), 2, 8,true);
            ft2->putText(image, "电灯连接正确:1分", Point(100, 300), 30, cv::Scalar(153, 51, 255), 2, 8,true);

            item["目前得分"]=res_score;
            item["最终得分"]=total_score;
            out.append(item);
            return success(out);

        }
         cv::imwrite(outputpath,image);
         //cout<<"name:"<<outputpath<<endl;

        imgIndex++;
        capture >> frame;
        return success(out);
    }
    return 0;
    

    
    
}


Json::Value LogicalController::detect_seg(const Json::Value& param){
    
    
    auto session = get_current_session();
    if(session->request.body.empty())
        return failure("Request body is empty!!!");

    // if base64
    // iLogger::base64_decode();
   
    //auto session = get_current_session();
	//auto image_data = iLogger::base64_decode(session->request.body);
	//iLogger::save_file("base_decode.jpg", image_data);
   
	cv::Mat imdata(1, session->request.body.size(), CV_8U, (char*)session->request.body.data());
    
    cv::Mat image = cv::imdecode(imdata, 1);
    cv::Mat mask_seg=image.clone();
    g_col=image.cols;
    g_row=image.rows;
    cout<<"g_col:"<<g_col<<endl;
    cout<<"g_row:"<<g_row<<endl;
    if(image.empty())
        return failure("Image decode failed!!!!");

    auto boxes = yolo_->commit(image).get();
    cout<<"box："<<boxes.size()<<endl;
    

    Json::Value out(Json::arrayValue);
    for(int i = 0; i < boxes.size(); ++i){
        auto& item = boxes[i];
        Json::Value itemj;
        itemj["left"] = item.left;
        itemj["top"] = item.top;
        itemj["right"] =item.right;
        itemj["bottom"] = item.bottom;
        itemj["class_label"] = item.class_label;
        itemj["confidence"] = item.confidence;
        out.append(itemj);
        
        

        //绘制image box
        uint8_t b, g, r;
        tie(b, g, r) = random_color(item.class_label+76);
        cv::rectangle(image, cv::Point(item.left, item.top), cv::Point(item.right, item.bottom), cv::Scalar(b, g, r), 2);
       

        auto name    = cocolabels[item.class_label];
        auto caption = cv::format("%s %.2f", name, item.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(item.left-3, item.top-20), cv::Point(item.left + 2*width/3, item.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(item.left, item.top-5), 0, 0.7, cv::Scalar::all(0), 2, 16);
        

        //绘制mask 
        mask_seg(cv::Rect(item.left,item.top,item.right-item.left,item.bottom-item.top)).setTo(cv::Scalar(b, g, r), item.mask);

    }
    addWeighted(image, 0.6, mask_seg, 0.5, 0, image); //将mask加在原图上面 
    imwrite("out.jpg",image);
    return success(out);
}

Json::Value LogicalController::getReturn(const Json::Value& param){

	Json::Value data;
	data["key"] = "aiVideo";
	data["video_url"] = "http://192.168.215.46/deepblue/aiVideo/person/scheduling/count?regionId=1";
	
	return success(data);
}
Json::Value LogicalController::getRes(const Json::Value& param){
    auto session = get_current_session();
	string json = session->request.body;
	string ff = ccutil::repstr(json, "?", "");
	auto item = Json::parse_string(ff);
	// int deviceId = item["deviceId"].asInt();
    // cout<<"cbdhcbhfdc:"<< deviceId <<endl;
    // string aiVideo=item["data"]["aiVideo"].asString();
    // string taskId=item["data"]["taskId"].asString();
    // int p1=item["data"]["point1"].asInt();
    // int p2=item["data"]["point2"].asInt();
    // int p3=item["data"]["point3"].asInt();
    // int p4=item["data"]["point4"].asInt();
    // int p5=item["data"]["point5"].asInt();
    // int cp=item["data"]["current_score"].asInt();
    // int fp=item["data"]["final_score"].asInt();
    string aiVideo=item["aiVideo"].asString();
    string taskId=item["taskId"].asString();
    int p1=item["pointOne"].asInt();
    int p2=item["pointTwo"].asInt();
    int p3=item["pointThree"].asInt();
    int p4=item["pointFour"].asInt();
    int p5=item["pointFive"].asInt();
    int cp=item["current_score"].asInt();
    int fp=item["finalScore"].asInt();
    INFO("得分点1:%i\n",p1);
    INFO("得分点2:%i\n",p2);
    INFO("得分点3:%i\n",p3);
    INFO("得分点4:%i\n",p4);
    INFO("得分点5:%i\n",p5);
    INFO("目前得分:%i\n",cp);
    INFO("最终得分:%i\n",fp);
    std::cout<<"taskid:"<<taskId<<std::endl;

    // cout<<"视频下载路径:"<<aiVideo<<endl;
    // cout<<"学生ID:"<<taskId<<endl;

	return success();
}
Json::Value LogicalController::putBase64Image(const Json::Value& param){
    
   

	/**
	 * 注意，这个函数的调用，请用工具（postman）以提交body的方式(raw)提交base64数据
	 * 才能够在request.body中拿到对应的base64，并正确解码后储存
	 * 1. 可以在网页上提交一个图片文件，并使用postman进行body-raw提交，例如网址是：https://base64.us/，选择页面下面的“选择文件”按钮
	 * 2. 去掉生成的base64数据前缀：data:image/png;base64,。保证是纯base64数据输入
	 *   这是一个图像的base64案例：iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsQAAA7EAZUrDhsAAABLSURBVEhLY2RY9OI/Ax0BE5SmG6DIh/8DJKAswoBxwwswTXcfjlpIdTBqIdXBqIVUB8O/8B61kOpg1EKqg1ELqQ5GLaQ6oLOFDAwA5z0K0dyTzgcAAAAASUVORK5CYII=
	 *   提交后能看到是个天蓝色的背景加上右上角有黄色的正方形
	 */

	auto session = get_current_session();
	auto image_data = iLogger::base64_decode(session->request.body);
    cout<<"22...."<<endl;
	iLogger::save_file("base_decode111.jpg", image_data);
	return success();
}
Json::Value LogicalController::getFile(const Json::Value& param){

	auto session = get_current_session();
	session->response.write_file("k12_test4_cut.mp4");//k12_test4_cut.mp4
	return success();
}
Json::Value LogicalController::getImg(const Json::Value& param){

	auto session = get_current_session();
	session->response.write_file("out.jpg");
	return success();
}

Json::Value LogicalController::getBinary(const Json::Value& param){

	auto session = get_current_session();
	auto data = iLogger::load_file("out.jpg");
	session->response.write_binary(data.data(), data.size());
	session->response.set_header("Content-Type", "image/jpeg");
	return success();
}


bool LogicalController::startup(){
    yolo_ = Yolo::create_infer("yolov5s_seg.trtmodel", Yolo::Type::V5, 0, 0.45F, 0.45);
    cls_ = create_engine_pool("newlight_cls.trtmodel",0);
    return yolo_ != nullptr && cls_ != nullptr;
}

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
static bool build_model(){

    //分割模型
    if(exists("yolov5s_seg.trtmodel")){
        printf("yolov5s-seg.trtmodel has exists.\n");
        return true;
    }


    //SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        1,
        "newnew-seg.onnx",
        "newnew-seg.trtmodel"
    );

    
    return true;
}
static bool build_model2()
{
    if(exists("newlight_cls.trtmodel")){
        printf("分类模型 has exists.\n");
        return true;
    }
    TRT::compile(
            TRT::Mode::FP32,
            1,
            "new_cls.onnx",
            "newlight_cls.trtmodel"
        );
    INFO("Done.");
    return true;
}

int start_http(int port = 8090){

    INFO("Create controller");
	auto logical_controller = make_shared<LogicalController>();
	if(!logical_controller->startup()){
		INFOE("Startup controller failed.");
		return -1;
	}

	string address = iLogger::format("0.0.0.0:%d", port);
	INFO("Create http server to: %s", address.c_str());

	auto server = createHttpServer(address, 32);
	if(!server)
		return -1;
    
    server->verbose();

	INFO("Add controller");
	server->add_controller("/api", logical_controller);

    // 这是一个vue的项目
	// server->add_controller("/", create_redirect_access_controller("./web"));
	// server->add_controller("/static", create_file_access_controller("./"));
	INFO("Access url: http://%s", address.c_str());

	INFO(
		"\n"
		"访问如下地址即可看到效果:\n"
		"1. http://%s/api/detec_seg              使用自定义写出内容作为response\n"
        "2. http://%s/api/getReturn              使用函数返回值中的json作为response\n"
        "3. http://%s/api/detect_seg_ai              使用函数返回值中的json作为response\n"
        "4. http://%s/api/getFile                使用自定义写出文件路径作为response\n",
		address.c_str(),address.c_str(),address.c_str(),address.c_str()
	);

	INFO("按下Ctrl + C结束程序");
	return iLogger::while_loop();
}

template<typename _T>
void worker(_T engine, cv::Mat image){

    typedef decltype(engine->commit(image)) futtype;
    vector<futtype> futs;
    for(int i = 0; i < 10; ++i){
        futs.emplace_back(engine->commit(image));
    }

    for(int i = 0; i < futs.size(); ++i){
        futs[i].get();
    }
}



int main(){

    // 新的实现
    if(!build_model()){
        return -1;
    }
    if(!build_model2()){
        return -1;
    }
    
    return start_http();
}