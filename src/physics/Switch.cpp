#include"Switch.h"

//欧式距离
float EuclideanDistance(vector<int>p1,vector<int>p2) //p=[label,x,y]
{
    int p11,p21,p12,p22;
    p11=p1[1];
    p21=p2[1];
    p12=p1[2];
    p22=p2[2];

    return  sqrt(pow((p11-p21),2)+pow((p12-p22),2));

}

//判断计算开关的距离
vector<float> SwitchDistance(vector<vector<int>>box)
{
    vector<float> res(2,0);//初始化 一个数组  0,0
    int i=0;
    while(i<box.size()-1)
    {
        for(int k=i+1;k<box.size();k++)
        {
            if((box[i][0]==3 && box[k][0]==2) ||(box[i][0]==2 && box[k][0]==3))
            {
                float D1=EuclideanDistance(box[i],box[k]);
                printf("D1 正极：%f \n",D1);
                res[0]=D1;
            }

            if((box[i][0]==3 && box[k][0]==1) ||(box[i][0]==1 && box[k][0]==3))
            {
                float D2=EuclideanDistance(box[i],box[k]);
                printf("D2 负极：%f \n",D2);
                res[1]=D2;
            }
            
        }
        i=i+1;
    }

    return res;

}

//判断状态
bool SwitchState(vector<float>distance)
{
    float d1 = distance[0];
    float d2 = distance[1];
    cout<<"d1:"<<d1<<endl;
    cout<<"d2:"<<d2<<endl;

    if(d1<d2)
    {
        printf("switch处于断开状态!\n");
        return false;
    }
    else if(d1>d2)
    {
        printf("switch处于闭合状态!\n");
        return true;
    }
    else
    {
        //d1==d2
        return NULL;

    }
}


