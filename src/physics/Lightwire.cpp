#include"Lightwire.h"

float WireLight(vector<vector<int>> wire,vector<vector<int>>light)
{
    //计算个条直线端点到电灯端点的距离
    vector<float>temp;
    vector<float>temp2;
    for(int i = 0; i < wire.size(); i++)
    {
        //cout<<"i:"<<i<<endl;
        vector<float>row1;
        vector<float>row2;

        for(int j = 0; j < light.size(); j++)
        {
            if(light[j][0] == 6) // 正极
            {
                float dis1=edudis(wire[i],light[j]);
                //cout<<"+ dis1:"<<dis1<<endl;
                row1.push_back(dis1);
            }
            
            if(light[j][0] == 7) //负极
            {
                float dis2=edudis(wire[i],light[j]);
                //cout<<"- dis2:"<<dis2<<endl;
                row2.push_back(dis2);
            }
        }
        //排序 小->大 结果
        sort(row1.begin(),row1.end());
        sort(row2.begin(),row2.end());
        temp.push_back(row1[0]);
        temp2.push_back(row2[0]);

    }

    //排序结果
    sort(temp.begin(),temp.end());
    sort(temp2.begin(),temp2.end());
    cout<<"light正极："<<temp[0]<<endl;
    cout<<"light负极："<<temp2[0]<<endl;
    //优化判断逻辑
    if(temp[0]<50 && temp2[0]<50)
    {
        printf("电灯连接正确！\n");
        return temp[0];
    }
    else
    {
        printf("电灯未被正常连接！\n");
        return NULL;
    }
}