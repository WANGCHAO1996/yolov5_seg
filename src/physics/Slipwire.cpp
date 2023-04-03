#include"Slipwire.h"


float edudis(vector<int>wire,vector<int>slip) //wire [x,y]  slip[label,x,y]
{
     int p11,p21,p12,p22;
     p11=wire[0];
     p12=wire[1];
     p21=slip[1];
     p22=slip[2];

    return  sqrt(pow((p11-p21),2)+pow((p12-p22),2));

}
float DisWireSlip(vector<vector<int>>wire,vector<vector<int>>slip)  //slip: label, 中心点x, 中心点y
{
    //计算个条直线端点到滑阻器端点的距离
    vector<float>temp;
    vector<float>temp2;
    for(int i = 0; i < wire.size(); i++)
    {
        //cout<<"i:"<<i<<endl;
        vector<float>row1;
        vector<float>row2;

        for(int j = 0; j < slip.size(); j++)
        {
            if(slip[j][0]==13) // 正极
            {
                float dis1=edudis(wire[i],slip[j]);
                //cout<<"+ dis1:"<<dis1<<endl;
                row1.push_back(dis1);
            }
            
            if(slip[j][0]==12) //负极
            {
                float dis2=edudis(wire[i],slip[j]);
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
    cout<<"滑阻器正极："<<temp[0]<<endl;
    cout<<"滑阻器负极："<<temp2[0]<<endl;
    cout<<"滑阻器正极："<<temp[1]<<endl;
    cout<<"滑阻器负极："<<temp2[1]<<endl;


    //输出

    if(temp[0]<30 && temp2[0]<30)//限制最小距离 正负
    {
        if(temp[0]<temp2[1] && temp2[0]<=temp[1])//附加条件：正1<负2 负1<正2 ： 一正一负
        {
            printf("滑阻器是一正一负连接！\n");
            return temp[0];
        }
        else //如果不满足：说明最近的两个值属于一个极性 
        {
            printf("滑阻器一极多次连接！\n");
            return NULL;
        }
    }
    else//不满足 限制条件 说明没有连接上
    {
        printf("电源连接错误--未被接入电路\n");
        return NULL;
    }

}

vector<int> SlipPosLocal(vector<vector<int>>aa,vector<vector<int>>bb,float pos_temp) //wire  slip_pos  min=temp[0]
{
    vector<int> res_pos;
    for(int i=0;i<aa.size();i++)
    {
        vector<float>row1;
        vector<float>row2;
        for(int j = 0;j < bb.size();j++)
        {
            if(j == 0)
            {
                float dis1 = edudis(aa[i],bb[j]);
                row1.push_back(dis1);
            }
            if(j == 1)
            {
                float dis2=edudis(aa[i],bb[j]);
                row2.push_back(dis2);
            }
        }
        sort(row1.begin(),row1.end());
        sort(row2.begin(),row2.end());

        if(row1[0]==pos_temp)
        {
            res_pos.push_back(bb[0][1]);
            res_pos.push_back(bb[0][2]);
            cout<<"pos0:"<<bb[0][1]<<","<<bb[0][2]<<endl;
            return res_pos;
        }
        if(row2[0]==pos_temp)
        {
            res_pos.push_back(bb[1][1]);
            res_pos.push_back(bb[1][2]);
            cout<<"pos1:"<<bb[1][1]<<","<<bb[1][2]<<endl;
            return res_pos;
        }

    }

}

float MaxSlider(vector<vector<int>>slider,vector<int>pos) // label, x ,y    x,y 
{   
    //计算pos 和 sider距离
    float dis = edudis(pos,slider[0]);
    cout<<"阻值："<<dis<<endl;
    if(dis>=100)
    {
        printf("阻值在最大！\n");
        return dis;
    }
    else
    {
        printf("阻值未处于最大状态！\n");

        return NULL;
    }

   

}


