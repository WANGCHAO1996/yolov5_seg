#include"Cutlight.h"
Mat CutLight(vector<vector<int>>box,Mat img)
{
    vector<int>light_box_res(4,0);
    for(int i=0;i<box.size();i++)
    {
        if(box[i][0]==4)
        {
            light_box_res[0]=box[i][1];
            light_box_res[1]=box[i][2];
            light_box_res[2]=box[i][3];
            light_box_res[3]=box[i][4];
        }
    }
    Rect imgselect=Rect(light_box_res[0],light_box_res[1],light_box_res[2]-light_box_res[0],light_box_res[3]-light_box_res[1]);
    Mat proimage=img.clone();//复制原图
    proimage = proimage(imgselect);

    //imwrite("cutlight.jpg",proimage);
    return  proimage;






}