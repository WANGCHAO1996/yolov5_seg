#include"cls.hpp"
#include <queue>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

// using namespace cv;
// using namespace std;
class EnginePoolImpl : public EnginePool{
public:
    virtual ~EnginePoolImpl()
    {
        stop();
    } 
    bool build(const string& file , int gpuid){
        promise<bool> pro;
        run_=true;
        worker_=make_shared<thread>(std::bind(&EnginePoolImpl::worker,this,std::ref(pro),file,gpuid));

        bool startup_finished = pro.get_future().get();
        if(!startup_finished)
        {
            stop();
        }
        return startup_finished;

    }
    void stop()
    {
        if(!run_)
        {
            return;
        }
        run_ = false;
        cv_.notify_all();
        worker_->join();
        worker_.reset();
    }
    virtual shared_future<PredictResult> commit(const Mat& image) override
    {
        Job job;
        job.image=image;
        job.pro.reset(new promise<PredictResult>());
        {
            std::unique_lock<mutex>l(lock_);
            jobs_.push(job);
        }
        cv_.notify_one();
        return job.pro->get_future();
    }


protected:
    struct Job{
        shared_ptr<promise<PredictResult>>pro;
        Mat image;
    };


    void softmax(float* ptr,int count){
        float sum=0;
        for(int i=0;i<count;i++)
        {
            sum+=expf(ptr[i]);
        }
        for(int i=0;i<count;i++)
        {
            ptr[i]=expf(ptr[i])/sum;
        }
    }

    virtual void worker(promise<bool>& pro,string file,int gpuid)
    {
        TRT::set_device(gpuid);
        auto engine = TRT::load_infer(file);
        bool engine_finished = engine != nullptr;

        pro.set_value(engine_finished);
        if(!engine_finished)
        {
            INFO("Load engine %s to gpu: %d faild",file.c_str(),gpuid);
            return;
        }

        auto input = engine->input();
        auto output = engine->output();
        int max_batch_size=engine->get_max_batch_size();
        vector<Job> fetch_jobs;

        float mean[]={0.220,0.224,0.225};
        float std[]={0.485,0.456,0.406};
        while(run_)
        {
            std::unique_lock<mutex> l(lock_);
            cv_.wait(l,[&]{return !run_ || !jobs_.empty();});

            if(!run_)
                break;
            
            fetch_jobs.clear();
            for(int i=0;i<max_batch_size && !jobs_.empty();++i){
                fetch_jobs.emplace_back(jobs_.front());
                jobs_.pop();
            }
        
            int infer_batch_size = fetch_jobs.size();
            for(int i=0;i<infer_batch_size;++i)
            {
                auto& item = fetch_jobs[i];
                cvtColor(item.image,item.image,cv::COLOR_BGR2RGB);
                input->set_norm_mat(i,item.image,mean,std);

            }

            engine->forward(false);

            for(int i=0;i<infer_batch_size;++i)
            {
                auto& item=fetch_jobs[i];
                float* begin=output->cpu<float>(i);
                int count = output->channel();
                softmax(begin,count);
                PredictResult pr;
                pr.label = std::max_element(begin,begin+count)-begin;
                pr.confidence=begin[pr.label];
                item.pro->set_value(pr);
            }
        }
    }
private:
    atomic<bool> run_{false};
    condition_variable cv_;
    mutex lock_;
    queue<Job> jobs_;
    shared_ptr<thread> worker_;

};

shared_ptr<EnginePool>create_engine_pool(const string& file,int gpuid)
{
    shared_ptr<EnginePoolImpl> instance(new EnginePoolImpl());
    if(!instance->build(file,gpuid))
        instance.reset();
    return instance;
}

vector<string> load_labels(const char* file){
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()){
        printf("open %d failed.\n", file);
        return lines;
    }
    
    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}

