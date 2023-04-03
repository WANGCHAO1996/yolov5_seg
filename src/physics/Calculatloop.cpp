#include"Calculatloop.h"
#include <stdlib.h>
//函数定义

bool OverLap(vector<int>bbox1,vector<int>bbox2)
{
    int x01,y01,x02,y02;
    int x11,y11,x12,y12;
    float lx,ly,sax,sbx,say,sby;
    x01=bbox1[0];
    y01=bbox1[1];
    x02=bbox1[2];
    y02=bbox1[3];
    x11=bbox2[0];
    y11=bbox2[1];
    x12=bbox2[2];
    y12=bbox2[3];
 

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2); //绝对值
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2);
    sax = abs(x01 - x02);
    sbx = abs(x11 - x12);
    say = abs(y01 - y02);
    sby = abs(y11 - y12);


    if(lx <= (sax/2+sbx/2)  & ly <= (say/2+sby/2))
    {
       
        return true;
    }
    else
    {
       
        return false;
    }


}

// int CalculatingLoop(vector<vector<int>>box)
// {
//     int i=0;
//     int result=0;
//     while(i < box.size()-1)
//     {
//         //cout<<"i="<<i<<endl;
        
//         for(int k = i+1; k<box.size();k++)
//         {
//             //数组切片 截取
//             vector<int>::const_iterator First1 = box[i].begin() + 1;  // 找到第  2 个迭代器 （idx=1）
//             vector<int>::const_iterator Second1 = box[i].begin() + 5; // 找到第  5 个迭代器 （idx=4）的下一个位置 

//             vector<int>::const_iterator First2 = box[k].begin() + 1;  // 找到第  2 个迭代器 （idx=1）
//             vector<int>::const_iterator Second2 = box[k].begin() + 5; // 找到第  5 个迭代器 （idx=4）的下一个位置 
//             vector<int> array1;
//             vector<int> array2;
//             array1.assign(First1,Second1);//左闭右开
//             array2.assign(First2,Second2);
//             if(OverLap(array1,array2))
//             {
//                 result=result+1;
//             }

//         }
        
//         i=i+1;
        
//     }
//     printf("res:%i\n",result);
//     if(result==8)
//     {
//         printf("电路中形成回路!\n");
//         return result;
//     }
//     else
//     {
//         printf("未形成正确回路！\n");
//         return result;
//     }
    

// }

int CalculatingLoop(vector<vector<int>>box)
{
    int i=0;
    int result=0;
    while(i < box.size()-1)
    {
        //cout<<"i="<<i<<endl;
        
        for(int k = i+1; k<box.size();k++)
        {
            //判断一下标签情况：杜绝wire和wire之间的计算 杜绝非wire元器件之间
            if((box[i][0]==15 && box[k][0]!=15)||(box[i][0]!=15 && box[k][0]==15))
            {
                //数组切片 截取
                vector<int>::const_iterator First1 = box[i].begin() + 1;  // 找到第  2 个迭代器 （idx=1）
                vector<int>::const_iterator Second1 = box[i].begin() + 5; // 找到第  5 个迭代器 （idx=4）的下一个位置 

                vector<int>::const_iterator First2 = box[k].begin() + 1;  // 找到第  2 个迭代器 （idx=1）
                vector<int>::const_iterator Second2 = box[k].begin() + 5; // 找到第  5 个迭代器 （idx=4）的下一个位置 
                vector<int> array1;
                vector<int> array2;
                array1.assign(First1,Second1);//左闭右开
                array2.assign(First2,Second2);
                if(OverLap(array1,array2))
                {
                    result=result+1;
                }

            }
            
        }
        
        i=i+1;
        
    }
    //判断wire个数
    int wire_num=0;
    for(int w = 0; w < box.size(); w++)
    {
        if(box[w][0]==15)
        {
            wire_num++;
        }
    }

    printf("res:%i\n",result);
    printf("wire:%i\n",wire_num);
    cout<<"box:"<<box.size()<<endl;
    if(result==8 && wire_num==4)
    {
        printf("电路中形成回路!\n");
        return result;
    }
    // if(result==8)
    // {
    //     printf("电路中形成回路!\n");
    //     return result;
    // }
    else
    {
        printf("未形成正确回路！\n");
        return 0;
    }
    

}

