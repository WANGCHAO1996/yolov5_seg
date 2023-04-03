#include "yolo.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>
#include "d_yolo.h"


extern int g_col;
extern int g_row;


namespace Yolo{
    using namespace cv;
    using namespace std;
    

    const char* type_name(Type type){
        switch(type){
        case Type::V5: return "YoloV5";
        case Type::V3: return "YoloV3";
        case Type::V7: return "YoloV7";
        case Type::X: return "YoloX";
        default: return "Unknow";
        }
    }

    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            // 这里取min的理由是
            // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
            // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
            // **
            float scale = std::min(scale_x, scale_y);

            /**
            这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
            scale, 0, -scale * from.width * 0.5 + to.width * 0.5
            0, scale, -scale * from.height * 0.5 + to.height * 0.5
            
            这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
            例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
            S = [
            scale,     0,      0
            0,     scale,      0
            0,         0,      1
            ]
            
            P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
            P = [
            1,        0,      -scale * from.width * 0.5
            0,        1,      -scale * from.height * 0.5
            0,        0,                1
            ]

            T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
            T = [
            1,        0,      to.width * 0.5,
            0,        1,      to.height * 0.5,
            0,        0,            1
            ]

            通过将3个矩阵顺序乘起来，即可得到下面的表达式：
            M = [
            scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
            0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
            0,        0,                     1
            ]
            去掉第三行就得到opencv需要的输入2x3矩阵
            **/

            /* 
                 + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
                参考：https://www.iteye.com/blog/handspeaker-1545126
            */
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
            
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    static float iou(const Box& a, const Box& b){
        float cleft 	= max(a.left, b.left);
        float ctop 		= max(a.top, b.top);
        float cright 	= min(a.right, b.right);
        float cbottom 	= min(a.bottom, b.bottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top);
        float b_area = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top);
        return c_area / (a_area + b_area - c_area);
    }

    static BoxArray cpu_nms(BoxArray& boxes, float threshold){

        std::sort(boxes.begin(), boxes.end(), [](BoxArray::const_reference a, BoxArray::const_reference b){
            return a.confidence > b.confidence;
        });

        BoxArray output;
        output.reserve(boxes.size());

        vector<bool> remove_flags(boxes.size());
        for(int i = 0; i < boxes.size(); ++i){

            if(remove_flags[i]) continue;

            auto& a = boxes[i];
            output.emplace_back(a);

            for(int j = i + 1; j < boxes.size(); ++j){
                if(remove_flags[j]) continue;
                
                auto& b = boxes[j];
                if(b.class_label == a.class_label){
                    if(iou(a, b) >= threshold)
                        remove_flags[j] = true;
                }
            }
        }
        return output;
    }

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,               // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const string& file, Type type, int gpuid, 
            float confidence_threshold, float nms_threshold,
            NMSMethod nms_method, int max_objects,
            bool use_multi_preprocess_stream
        ){
            if(type == Type::V5 || type == Type::V3 || type == Type::V7){
                normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            }else if(type == Type::X){
                //float mean[] = {0.485, 0.456, 0.406};
                //float std[]  = {0.229, 0.224, 0.225};
                //normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
                normalize_ = CUDAKernel::Norm::None();
            }else{
                INFOE("Unsupport type %d", type);
            }
            
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            nms_method_           = nms_method;
            max_objects_          = max_objects;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        virtual void worker(promise<bool>& result) override
        {

            string file = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            

            const int MAX_IMAGE_BBOX  = max_objects_;
            const int NUM_BOX_ELEMENT = 7+32;      // left, top, right, bottom, confidence, class, keepflag 32mask coefficient
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images"); //模型的输入 要和onnx模型对上
            auto output        = engine->tensor("output0"); //模型的输出 要和onnx模型对上
            auto output1       = engine->tensor("output1");

            int num_classes    = output->size(2) - 5 ; //模型类别数目
            //cout<<"numclass:"<<num_classes<<endl;

            input_width_       = input->size(3); //宽 960
            input_height_      = input->size(2); //高 544
            int segWidth       = input->size(3)/4; //seg 宽 240
            int segHeight       = input->size(2)/4; //seg 高 136
            float mask_thresh  = 0.35;
            int segchannel     = 32;
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();
            // int col=810;//810; //1920  2304
            // int row=1080;//1080; //1080  1296
            // Rect holeImgRect(0, 0, col, row);
            cout<<"1........"<<endl;

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size))
            {
                int col=g_col;
                int row=g_row;
                Rect holeImgRect(0, 0, col, row);
                

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();

                    if(mono->get_stream() != stream_){
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }

                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }
                engine->forward(false);
               
                //output1 ==>proto  另一个分支
                float* output_ptr = output1->cpu<float>();
                //vector 2 mat
                int size[]={segchannel,segHeight,segWidth};
                //cout<<"size"<<size[0]<<endl;
                cv::Mat mask_protos = cv::Mat_<float>(3,size,CV_8UC1);
                for(int iii=0;iii<segchannel;iii++)
                {   
                    //unchar *data=mask_protos.ptr<unchar>(iii);
                    for(int jjj=0;jjj<segHeight;jjj++)
                    {
                        //unchar *data2=data.ptr<unchar>(jjj);
                        for(int kkk=0;kkk<segWidth;kkk++)
                        {
                            //data2[kkk]=output_ptr[iii*136*240+jjj*240+kkk];
                            mask_protos.at<float>(iii,jjj,kkk)=output_ptr[iii*segHeight*segWidth+jjj*segWidth+kkk];
                        }
                    }
                }
                

                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);

                    //printf("aaa==a:%f\n",output_array_ptr);
                    if(nms_method_ == NMSMethod::FastGPU){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }

                output_array_device.to_cpu();

                //new a mat ,vector 
                Mat mask_proposals;
                vector<OutputSeg> f_output;
                vector<vector<float>>proposal;  //[23,32] output0  =>mask

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_boxes   = job.output;
                    for(int i = 0; i < count; ++i)
                    {
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT; //7
                        int label    = pbox[5];
                        int keepflag = pbox[6];
                        vector<float> temp;
                        OutputSeg result;

                        if(keepflag == 1){
                            
                            for(int ii=0;ii<segchannel;ii++)
                            {
                                temp.push_back(pbox[ii+7]);
                            }
                            proposal.push_back(temp);
                            
                            result.id=pbox[5];
                            result.confidence=pbox[4];
                            cv::Rect rect(pbox[0], pbox[1], pbox[2]-pbox[0], pbox[3]-pbox[1]);
                            result.box=rect & holeImgRect; //x,y,w,h
                            f_output.push_back(result);
                            
                            //image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label,mask_proposals,mask_protos);
                        }
                        
                    }
                    //转 mat 
                    for (int i = 0; i < proposal.size(); ++i)
                    {
                        mask_proposals.push_back(Mat(proposal[i]).t());
                    }
                    // cout<<mask_proposals<<endl;
                    
                    //计算params

                    Vec4d params;  //根据实际图片输入 和 onnx模型输入输出 计算的，此处直接写死
                    float r =std::min((float)input->size(3)/(float)col,(float)input->size(2)/(float)row);//模型输入尺寸/图片真实尺寸
                    //cout<<"r:"<<r<<endl;
                    float ratio[2]{r,r};
                    int newUnpad[2]{(int)std::round((float)col * r),(int)std::round((float)row * r)};
                    auto dw = (float)(input->size(3) - newUnpad[0]);//960-960 = 0.0
	                auto dh = (float)(input->size(2) - newUnpad[1]);//544-540 =4.0
                    dw /= 2.0f;  //0
	                dh /= 2.0f; //2
                    int top = int(std::round(dh - 0.1f));  //2.0-0.1f
                    int bottom = int(std::round(dh + 0.1f)); //2.0+0.1f
                    int left = int(std::round(dw - 0.1f)); //round 四舍五入取整数 round(-0.1)
                    int right = int(std::round(dw + 0.1f)); // round(0.1)
    
                    params[0] = ratio[0]; //0.5
                    params[1] = ratio[1]; //0.5
                    params[2] = left;// int(std::round) 0
                    params[3] = top; // 2.0
                    cout<<"0:"<<params[0]<<endl;
                    cout<<"1:"<<params[1]<<endl;
                    cout<<"2:"<<params[2]<<endl;
                    cout<<"3:"<<params[3]<<endl;

                    // params[0]=0.59259;//0.5;
                    // params[1]=0.59259;//0.5
                    // params[2]=80;//0.0;
                    // params[3]=0;//2.0;


                    //逻辑 GetMask
                    
                    Mat protos = mask_protos.reshape(0, {segchannel,segHeight * segWidth});
                    Mat matmulRes = ( mask_proposals * protos).t(); //23,32 * 32,32640 ==> 23,32640
                    Mat masks = matmulRes.reshape(proposal.size(),{segHeight,segWidth}); //上一步骤作转置的原因：//Mat Mat::reshape(int cn,int rows=0) const cn:表示通道数（channels），如果设置为0，则表示通道不变；
                    vector<Mat> maskChannels; //分离通道
                    split(masks, maskChannels);
                    
                    for (int index = 0; index < f_output.size(); ++index) {//23
                        Mat dest,mask;
                        //sigmoid
                        cv::exp(-maskChannels[index],dest);//e^x
                        dest= 1.0/(1.0 + dest);
                        // cout<<"111"<<endl;
                        // cout<<int(params[2] / input_width_ * segWidth)<<endl;
                        // cout<<int(params[3] / input_height_ * segHeight)<<endl;
                        // cout<<int(segWidth - params[2] / 2)<<endl;
                        // cout<<int(segHeight+1- params[3]/2)<<endl;
                        //_netWidth = 960; _netHeight=544;  //ONNX图片输入宽度\高度  //	const int _segWidth = 240;
                        Rect roi(int(params[2] / input_width_ * segWidth), int(params[3] / input_height_ * segHeight), int(segWidth - params[2] / 2), int(segHeight- params[3]/2)); //136-params[3]/2最后一个参数改了 mask会有偏移
                        dest = dest(roi);
                        resize(dest, mask, cv::Size(col,row), INTER_NEAREST);//srcImgShape （1920，1080）//INTER_NEAREST 最近临插值  PYTHON中用的就是 INTER_LINEAR - 双线性插值
                        
                        //crop
                        Rect temp_rect = f_output[index].box;
                        mask = mask(temp_rect) > mask_thresh; //mask_threshg mask阈值
                        f_output[index].boxMask =mask;
                        int lf=f_output[index].box.x;
                        int tp=f_output[index].box.y;
                        int wd=f_output[index].box.width;
                        int hg=f_output[index].box.height;
                        

                        image_based_boxes.emplace_back(lf, tp, lf+wd, tp+hg, f_output[index].confidence, f_output[index].id,mask);
                    }
                    


                    if(nms_method_ == NMSMethod::CPU){
                        image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);
                    }
                    job.pro->set_value(image_based_boxes);
                }

                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{

            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            if(image.empty()){
                INFOE("Image is empty");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor->data();
            TRT::CUStream preprocess_stream = nullptr;

            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());

                if(use_multi_preprocess_stream_){
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));

                    // owner = true, stream needs to be free during deconstruction
                    tensor->set_stream(preprocess_stream, true);
                }else{
                    preprocess_stream = stream_;

                    // owner = false, tensor ignored the stream
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);
            
            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image      = image.cols * image.rows * 3;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            //checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 114, 
                normalize_, preprocess_stream
            );
            return true;
        }

        virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<BoxArray> commit(const Mat& image) override{
            // int g_col=image.cols;
            // int g_row=image.rows;
            // cout<<"2..."<<endl;
            // cout<<g_col<<endl;
            // cout<<g_row<<endl;
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        int max_objects_            = 1024;
        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
       
        
       
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid, 
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects,
        bool use_multi_preprocess_stream
    )
    {
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(
            engine_file, type, gpuid, confidence_threshold, 
            nms_threshold, nms_method, max_objects, use_multi_preprocess_stream)
        ){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch){

        CUDAKernel::Norm normalize;
        if(type == Type::V5 || type == Type::V3 || type == Type::V7){
            normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
        }else if(type == Type::X){
            //float mean[] = {0.485, 0.456, 0.406};
            //float std[]  = {0.229, 0.224, 0.225};
            //normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
            normalize = CUDAKernel::Norm::None();
        }else{
            INFOE("Unsupport type %d", type);
        }
        
        Size input_size(tensor->size(3), tensor->size(2));
        AffineMatrix affine;
        affine.compute(image.size(), input_size);
        //cout<<"size:"<<image.size()<<endl;

        size_t size_image      = image.cols * image.rows * 3;
        size_t size_matrix     = iLogger::upbound(sizeof(affine.d2i), 32);
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device         = size_matrix + gpu_workspace;

        uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
        float* affine_matrix_host     = (float*)cpu_workspace;
        uint8_t* image_host           = size_matrix + cpu_workspace;
        auto stream                   = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

        CUDAKernel::warp_affine_bilinear_and_normalize_plane(
            image_device,               image.cols * 3,       image.cols,       image.rows, 
            tensor->gpu<float>(ibatch), input_size.width,     input_size.height, 
            affine_matrix_device, 114, 
            normalize, stream
        );
        tensor->synchronize();
    }
};