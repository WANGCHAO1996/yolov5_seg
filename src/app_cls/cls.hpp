#include <future>
#include <thread>
#include <condition_variable>
#include <queue>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <infer/trt_infer.hpp>
#include<fstream>


using namespace cv;
using namespace std;

struct PredictResult{
    int label;
    float confidence;
};
class EnginePool{
public:
    virtual shared_future<PredictResult>commit(const Mat& image)=0;
};
shared_ptr<EnginePool>create_engine_pool(const string& file,int gpuid);

vector<string> load_labels(const char* file);


