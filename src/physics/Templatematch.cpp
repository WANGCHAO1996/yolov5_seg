#include"Templatematch.h"


//定义模式匹配函数
vector<vector<int>> TemplateMatch(Mat img, int m)
{
    Mat img0;
    //threshold(img,img0,0,255,THRESH_BINARY|THRESH_OTSU);
    //imwrite("skeleton33.jpg",img0);
    //设置匹配模板 mask0->mask3
    vector<vector<int>> mask0(m,vector<int>(m,0));
    vector<vector<int>> mask1(m,vector<int>(m,0));
    vector<vector<int>> mask2(m,vector<int>(m,0));
    vector<vector<int>> mask3(m,vector<int>(m,0));
    mask0[0][1]=255;
    mask1[1][0]=255;
    mask2[2][1]=255;
    mask3[1][2]=255;

    //ms 模板集合
    vector<vector<vector<int>>>ms;
    ms.push_back(mask0);
    ms.push_back(mask1);
    ms.push_back(mask2);
    ms.push_back(mask3);

    //临时图像
    Mat temp_img=img.clone();
    //vector<vector<int>> matrix;
    Mat matrix111; 
    Mat post_matrix;

    //边界补零处理  待定
    int row = m; 
    int col = m;
    int r = (row - 1) / 2;
    int c = (col - 1) / 2;

    // cout<<"rows:"<<img.cols<<endl;1080

    //存储结果
    vector<vector<int>> res;
    vector<int> temp_res(2,0);
    for(int i = r; i < img.rows - r; i++)  //高 1080
    { 
        for(int j = c; j < img.cols - c ; j++) //宽 1920
        {
            //切片继续
            matrix111=img(Rect(j-c,i-r,3,3));//以（i,j）为中心点的3*3的Mat  Rect(x,y,w,h)

            //与mask的库进行匹配 循环
            for(int mm=0;mm<4;mm++) //ms的维度=4
            {   
                int pt_num=0;
                for(int ii=0;ii<3;ii++)
                {
                    for(int jj=0;jj<3;jj++)
                    {
                        
                        if(int(matrix111.at<uchar>(ii,jj)) == ms[mm][ii][jj])
                        {
                           
                           pt_num++; 

                        } 
                       
                    }
                }
                if(pt_num == 9) //九个点都匹配上了
                {
                    // cout<<"打印："<<matrix111.at<int>(0,0)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(0,1)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(0,2)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(1,0)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(1,1)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(1,2)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(2,0)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(2,1)<<endl;
                    // cout<<"打印："<<matrix111.at<int>(2,2)<<endl;
                    if(mm==0)
                    {
                        printf("第一类导线头的坐标为：%i,%i \n",i-r,j-c+1);
                        //排除干扰点 以该点为模板中心 i=i-c j=j-r+1
                        post_matrix=img(Rect(j-2*c+1,i-2*r,3,3));//
                        //计算像素和
                        float sum1=mat_sum(post_matrix);
                        if(sum1 <= (2*255))
                        {
                            temp_res[1]=i-r;
                            temp_res[0]=j-c+1;
                            res.push_back(temp_res);
                        }
                        else{
                            printf("去除干扰点\n");
                        }
                   }
                    if(mm==1)
                    {
                        printf("第二类导线头的坐标为：%i,%i \n",i,j-c);
                        //排除干扰点 以该点为模板中心 i=i j=j-r
                        post_matrix=img(Rect(j-2*c,i-r,3,3));
                        //计算像素和
                        float sum2=mat_sum(post_matrix);
                        if(sum2 <= (2*255))
                        {
                            temp_res[1]=i;
                            temp_res[0]=j-c;
                            res.push_back(temp_res);
                        }
                        else{
                            printf("去除干扰点\n");
                        }
                    }

                    if(mm==2)
                    {
                        printf("第三类导线头的坐标为：%i,%i \n",i+1,j-c+1);
                        //排除干扰点 以该点为模板中心 i=i+1 j=j-r+1
                        post_matrix=img(Rect(j-2*c+1,i-r+1,3,3));
                        //计算像素和
                        float sum3=mat_sum(post_matrix);
                        if(sum3 <= (2*255))
                        {
                            temp_res[1]=i+1;
                            temp_res[0]=j-c+1;
                            res.push_back(temp_res);
                        }
                        else{
                            printf("去除干扰点\n");
                        }
                    }

                    if(mm==3)
                    {
                        printf("第四类导线头的坐标为：%i,%i \n",i,j+1);
                        //排除干扰点 以该点为模板中心 i=i j=j+1  matrix111=img(Rect(j-c,i-r,3,3))
                        post_matrix=temp_img(Rect(j,i-1,3,3));
                        // //计算像素和
                        float sum4=mat_sum(post_matrix);
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(0,0))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(0,1))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(0,2))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(1,0))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(1,1))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(1,2))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(2,0))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(2,1))<<endl;
                        // // cout<<"打印："<<int(post_matrix.at<uchar>(2,2))<<endl;

                        if(sum4 <= (2*255.0))
                        {
                            temp_res[1]=i;
                            temp_res[0]=j+1;
                            res.push_back(temp_res);
                        }
                        else{
                            printf("去除干扰点\n");
                        }
                    }
                
                
                }
            }
        }
    }
    //去重复 res中存在重复的结果
    res=DuplicateRemoval(res);
    return res;

}

int mat_sum(Mat mat)
{
    int col = mat.cols;
    int row = mat.rows;
    int res = 0;

    for( int i=0;i<3;i++)
    {
        for( int j=0;j<3;j++)
        {
        
            res+=int(mat.at<uchar>(j,i));
        }

    }
    //cout<<"res:"<<res<<endl;
    return res;
}
vector<vector<int>>DuplicateRemoval(vector<vector<int>> r)
{
    //排序
    sort(r.begin(),r.end());
    r.erase(unique(r.begin(),r.end()),r.end());
    printf("wire端点去重复....\n");

    print_vec2(r);
    return r;
}

void print_vec2(vector<vector<int>>& vec)
{
    for (auto v : vec) {
        for (auto i : v) cout << i << " ";
        cout << endl;
    }
}