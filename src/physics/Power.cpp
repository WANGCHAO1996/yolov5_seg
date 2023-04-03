#include"Power.h"


float WirePower(vector<vector<int>> wire,vector<vector<int>>power) //wire :x,y  power label,x,y 
{
     //计算个条直线端点到电源端点的距离
    vector<float>temp;
    vector<float>temp2;
    for(int i = 0; i < wire.size(); i++)
    {
        //cout<<"i:"<<i<<endl;
        vector<float>row1;
        vector<float>row2;

        for(int j = 0; j < power.size(); j++)
        {
            if(power[j][0]==10) // 正极
            {
                float dis1=edudis(wire[i],power[j]);
                //cout<<"+ dis1:"<<dis1<<endl;
                row1.push_back(dis1);
            }
            
            if(power[j][0]==9) //负极
            {
                float dis2=edudis(wire[i],power[j]);
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
    cout<<"电源正极："<<temp[0]<<endl;
    cout<<"电源负极："<<temp2[0]<<endl;
    //top2
    cout<<"电源正极："<<temp[1]<<endl;
    cout<<"电源负极："<<temp2[1]<<endl;

    //优化判断逻辑
    if(temp[0]<50 && temp2[0]<50)
    {
        if(temp[1]>=50 && temp2[1]>=50)
        {
            printf("电源是一正一负连接！\n");
            return temp[0];
        }
        else
        {
            printf("电源被多次连接！\n");
            return NULL;
        }
    }
    else
    {
        printf("电源连接错误--未被接入电路\n");
        return NULL;
    }

}