#include"Cutobj.h"
Mat CutObj(vector<vector<int>>box,Mat img,int key)
{
    vector<int>obj_res(4,0);
    
    if(key=0)
    {
        for(int i=0;i<box.size();i++)
        {
            if(box[i][0]==0)
            {
                obj_res[0]=max(box[i][1]-30,0);
                obj_res[1]=max(box[i][2]-30,0);
                obj_res[2]=min(box[i][3]+30,img.cols);
                obj_res[3]=min(box[i][4]+30,img.rows);
            }
        }
        Rect imgselect=Rect(obj_res[0],obj_res[1],obj_res[2]-obj_res[0],obj_res[3]-obj_res[1]);
        Mat proimage=img.clone();//复制原图
        proimage = proimage(imgselect);
        return  proimage;
    }
    if(key=1)
    {
        for(int i=0;i<box.size();i++)
        {
            if(box[i][0]==8)
            {
                obj_res[0]=max(box[i][1]-30,0);
                obj_res[1]=max(box[i][2]-30,0);
                obj_res[2]=min(box[i][3]+30,img.cols);
                obj_res[3]=min(box[i][4]+30,img.rows);
            }
        }
        Rect imgselect=Rect(obj_res[0],obj_res[1],obj_res[2]-obj_res[0],obj_res[3]-obj_res[1]);
        Mat proimage=img.clone();//复制原图
        proimage = proimage(imgselect);
        return  proimage;
    }
    if(key=2)
    {
        for(int i=0;i<box.size();i++)
        {
            if(box[i][0]==11)
            {
                obj_res[0]=max(box[i][1]-30,0);
                obj_res[1]=max(box[i][2]-30,0);
                obj_res[2]=min(box[i][3]+30,img.cols);
                obj_res[3]=min(box[i][4]+30,img.rows);
            }
        }
        Rect imgselect=Rect(obj_res[0],obj_res[1],obj_res[2]-obj_res[0],obj_res[3]-obj_res[1]);
        Mat proimage=img.clone();//复制原图
        proimage = proimage(imgselect);
        return  proimage;
    }
    if(key=3)
    {
        for(int i=0;i<box.size();i++)
        {
            if(box[i][0]==15)
            {
                obj_res[0]=max(box[i][1]-30,0);
                obj_res[1]=max(box[i][2]-30,0);
                obj_res[2]=min(box[i][3]+30,img.cols);
                obj_res[3]=min(box[i][4]+30,img.rows);
            }
        }
        Rect imgselect=Rect(obj_res[0],obj_res[1],obj_res[2]-obj_res[0],obj_res[3]-obj_res[1]);
        Mat proimage=img.clone();//复制原图
        proimage = proimage(imgselect);
        return  proimage;
    }

    
    //imwrite("cutlight.jpg",proimage);
    






}