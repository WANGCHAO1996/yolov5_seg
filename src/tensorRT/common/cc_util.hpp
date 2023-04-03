
#ifndef CC_UTILS_HPP
#define CC_UTILS_HPP

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif

#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <future>
#include <thread>
#include <condition_variable>

#define CCUtilVersion  "1.1.0"

namespace ccutil {

#define ORG_Center			-1000000
#define Assert(op)			ccutil::AssertStream(!(!(op)), __FILE__, __LINE__, #op)
#define AssertEQ(a, b)		Assert(a == b)
#define AssertNE(a, b)		Assert(a != b)
#define AssertLE(a, b)		Assert(a <= b)
#define AssertLT(a, b)		Assert(a < b )
#define AssertGE(a, b)		Assert(a >= b)
#define AssertGT(a, b)		Assert(a > b )
#define LINFO				0
#define LWARNING			1
#define LERROR				2
#define LFATAL				3
#define INFO(...)			ccutil::__log_func(__FILE__, __LINE__, LINFO, __VA_ARGS__)
#define INFOW(...)			ccutil::__log_func(__FILE__, __LINE__, LWARNING, __VA_ARGS__)
#define INFOE(...)			ccutil::__log_func(__FILE__, __LINE__, LERROR, __VA_ARGS__)
#define INFOF(...)			ccutil::__log_func(__FILE__, __LINE__, LFATAL, __VA_ARGS__)
#define LOG(level)			ccutil::LoggerStream(true, level, __FILE__, __LINE__)
#define LOG_IF(level, op)	ccutil::LoggerStream(!(!(op)), level, __FILE__, __LINE__)

	using std::string;
	using std::vector;
	using std::map;
	using std::unordered_map;

	struct BBox {

		float score = 0; float x = 0; float y = 0; float r = 0; float b = 0;
		int label = 0;

		BBox();
		BBox(const cv::Rect& other);
		BBox(float x, float y, float r, float b, float score = 0, int label = 0);
		float width() const;
		float height() const;
		float area() const;
		cv::Point2f center() const;
		float iouOf(const BBox& b) const;
		float iouMinOf(const BBox& b) const;
		BBox mergeOf(const BBox& b) const;
		BBox expandMargin(float margin, const cv::Size& limit = cv::Size()) const;
		BBox expand(float ratio, const cv::Size& limit = cv::Size()) const;
		cv::Rect box() const;
		operator cv::Rect() const;
		cv::Point tl() const;
		cv::Point rb() const;
		BBox offset(const cv::Point& position) const;
		BBox transfrom(cv::Size sourceSize, cv::Size dstSize);
	};

	struct LabBBox : BBox {
		string filename; string classname;
	};

	string tostr(int val);
	string tostr(unsigned int val);
	string tostr(long val);
	string tostr(unsigned long val);
	string tostr(long long val);
	string tostr(unsigned long long val);
	string tostr(long double val);
	string tostr(double val);
	string tostr(float val);
	inline const string& tostr(const string& val) { return val; }
	inline string& tostr(string& val) { return val; }
	inline string tostr(const char* val) { return val; }
	inline string tostr(char* val) { return val; }
	inline string tostr(const void* val);

	string format(const char* fmt, ...);
	typedef bool(*LoggerListener)(const char* file, int line, int level, const char* message);

	struct Stream {
		string msg_;

		template <typename _T>
		Stream& operator << (const vector<_T>& list) {
			if (list.empty()) {
				msg_ += "empty list.";
				return *this;
			}

			msg_ += format("list[%d] = {", list.size());
			for (int i = 0; i < list.size(); ++i) {

				if (i < (int)list.size() - 1) {
					msg_ += "\"" + tostr(list[i]) + "\",";
				}
				else {
					msg_ += "\"" + tostr(list[i]) + "\"}";
				}
			}
			return *this;
		}

		template <typename _T>
		Stream& operator << (const _T& v) {
			msg_ += tostr(v);
			return *this;
		}
	};

	struct AssertStream : public Stream {
		bool condition_;
		const char* file_;
		int line_;
		const char* code_;

		AssertStream(bool condition, const char* file, int line, const char* code);
		AssertStream();
		virtual ~AssertStream();
	};

	struct LoggerStream : public Stream {

		bool condition_;
		const char* file_;
		int line_;
		int level_;

		LoggerStream(bool condition, int level, const char* file, int line);
		virtual ~LoggerStream();
	};

	struct Timer {
		double tick = 0;

		Timer();
		void begin();
		double end();	//ms
	};

	struct GenNumber {
		volatile int next_ = 0;

		GenNumber() {}
		GenNumber(int start):next_(start) {}
		int next();
	};

	inline vector<int> range(int begin, int end, int step = 1) {
		vector<int> out;
		for (int i = begin; i != end; i += step)
			out.push_back(i);
		return out;
	}

	inline vector<int> range(int end) {
		vector<int> out;
		for (int i = 0; i != end; ++i)
			out.push_back(i);
		return out;
	}

	template<typename _T>
	struct __SelectElement {
		_T& ref_;
		__SelectElement(_T& v):ref_(v) {}

		typename _T::const_reference operator[](int pos) const {
			if (pos < 0)
				pos = (int)ref_.size() + pos;
			return ref_[pos];
		}

		typename _T::reference operator[](int pos) {
			if (pos < 0)
				pos = (int)ref_.size() + pos;
			return ref_[pos];
		}

		_T operator()(int end) {
			if (end < 0)
				end = (int)ref_.size() + end + 1;

			_T out;
			for (int index = 0; index != end; ++index)
				out.push_back(ref_[index]);
			return std::move(out);
		}

		_T operator()(int begin, int end, int step = 1) {
			if (end < 0)
				end = (int)ref_.size() + end + 1;

			_T out;
			for (int index = begin; index != end; index += step)
				out.push_back(ref_[index]);
			return std::move(out);
		}

		vector<typename _T::value_type> operator[](const vector<int>& inds) {
			_T out;
			for (auto& index : inds)
				out.push_back(ref_[index]);
			return std::move(out);
		}
	};

	//vector<int> a = { 1, 2, 3 };
	//auto val = S(a)[-1];            ->  val = 3
	//auto list = S(a)(-1);			  ->  list = {1, 2},   0:-1    -----)   0:2(exclude 2)
	//auto list = S(a)(0, 3)		  ->  list = {1, 2, 3} 0:3     -----)   0:3(exclude 3)
	//auto list = S(a)(0, 3, 1)		  ->  list = {1, 2, 3} 0:3     -----)   0:3(exclude 3)

	//string text = "liuanqi____abc";
	//auto val = S(text)(0, -3);      ->  val = "liuanqi____"
	template<typename _T>
	__SelectElement<_T> S(_T& val) { return __SelectElement<_T>(val); }


	class ParameterParser{
	public:
		void parse(int argc, char** argv);

	private:
		unordered_map<string, string> kwargs_;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////
	time_t lastModify(const string& file);
	bool isFile(const string& file);
	bool isFolder(const string& file);
	string dateNow();
	string timeNow();
	string nowGMTime();
	string gmtime(time_t t);
	time_t gmTime2CTime(const string& gmt);
	void sleep(int ms);
	long long timestampNow();
	time_t string2time_t(string string_time);
	string nowFormat(const string& fmt);
	void setLoggerSaveDirectory(const string& loggerDirectory);
	void setLoggerListener(LoggerListener func);
	void setLogger(bool hasLogger = true);
	bool hasLogger();
	LoggerListener getCatchLoggerListener();
	void __log_func(const char* file, int line, int level, const char* fmt, ...);
	///////////////////////////////////////////////////////////////////////////////////////////////

	//bbox nms
	vector<BBox> nmsAsClass(const vector<BBox>& objs, float iou_threshold);
	vector<BBox> nms(vector<BBox>& objs, float iou_threshold);
	vector<BBox> nmsMinIoU(vector<BBox>& objs, float iou_threshold);

	//bbox softnms with linear method
	vector<BBox> softnms(vector<BBox>& B, float iou_threshold);

	//string operator
	vector<string> split(const string& str, const string& spstr);
	vector<int> splitInt(const string& str, const string& spstr);
	vector<float> splitFloat(const string& str, const string& spstr);

	//str="abcdef"   begin="ab"   end="f"   return "cde"
	string middle(const string& str, const string& begin, const string& end);

	//load bbox from std xml file
	vector<LabBBox> loadxmlFromData(const string& data, int* width, int* height, const string& filter);
	vector<LabBBox> loadxml(const string& file, int* width = nullptr, int* height = nullptr, const string& filter = "");
	bool savexml(const string& file, int width, int height, const vector<LabBBox>& objs);
	bool xmlEmpty(const string& file);
	bool xmlHasObject(const string& file, const string& classes);

	vector<string> loadList(const string& listfile);
	void rmblank(vector<string>& list);
	bool isblank(const string& str, char blank = ' ');
	bool saveList(const string& file, const vector<string>& list);

	//name,label,x,y,r,b
	//key = name, value = label,x,y,r,b
	map<string, string> loadListMap(const string& listfile);
	cv::Mat loadMatrix(FILE* file);
	bool saveMatrix(FILE* file, const cv::Mat& m);
	cv::Mat loadMatrix(const string& file);
	bool saveMatrix(const string& file, const cv::Mat& m);
	bool imwrite(const string& file, const cv::Mat& image);
	cv::Mat decodeImage(const string& data, int flags = 1);

	//voc
	string vocxml(const string& vocjpg);
	string vocjpg(const string& vocxml);

	//file operator
	//read file return all data
	string loadfile(const string& file);
	size_t fileSize(const string& file);
	bool savefile(const string& file, const string& data, bool mk_dirs = true);
	bool savefile(const string& file, const void* data, size_t length, bool mk_dirs = true);

	//file exists
	bool exists(const string& path);

	//filename
	//  c:/a/abc.xml
	//  include_suffix = true :   return abc.xml
	//  include_suffix = false :  return abc
	string fileName(const string& path, bool include_suffix = false);

	//  c:/abcdef   ->  return c:/
	//  c:/abcddd/  ->  return c:/abcddd/
	string directory(const string& path);

	//with
	bool beginsWith(const string& str, const string& with);
	bool endsWith(const string& str, const string& with);

	//replace suffix
	//abc.txt, xml           ->  abc.xml
	string repsuffix(const string& path, const string& newSuffix);
	string repstrFast(const string& str, const string& token, const string& value);
	string repstr(const string& str, const string& token, const string& value);

	//remove suffix
	//abc.txt               ->   abc
	//c:/asdf/aaa.txt       ->   c:/asdf/aaa
	string rmsuffix(const string& path);

	string md5(const void* data, int length);
	string md5(const string& data);

	//find 
	vector<string> findFiles(const string& directory, const string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);
	vector<string> findFilesAndCacheList(const string& directory, const string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

	//  a == A   ->   ignore_case=true   ->  return true
	bool alphabetEqual(char a, char b, bool ignore_case);

	//   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true
	bool patternMatch(const char* str, const char* matcher, bool igrnoe_case = true);

	vector<string> batchRepSuffix(const vector<string>& filelist, const string& newSuffix);

	template<typename _T, typename _Op>
	void each(_T& list, const _Op& op) { for (auto& item : list) op(item); }

	template<typename _T, typename _Op>
	void each_index(_T& list, const _Op& op) { for (int i = 0; i < list.size(); ++i) op(i, list[i]); }

#define cceach(_lst, _op)	do{ for (int _index = 0; _index < _lst.size(); ++_index) { auto& _item = _lst[_index]; {_op;}} }while(0);

	template<typename _T>
	void shuffle(_T& var) { std::random_shuffle(var.begin(), var.end()); }

	template<typename _T>
	void sameSizeArray(_T& a, _T& b) {
		if (a.size() > b.size())
			a.erase(a.begin() + b.size(), a.end());
		else if (a.size() < b.size())
			b.erase(b.begin() + a.size(), b.end());
	}

	//make dir and subdirs
	bool mkdirs(const string& path);
	bool mkdir(const string& path);
	bool copyTo(const string& src, const string& dst);
	bool moveTo(const string& src, const string& dst);
	bool rmtree(const string& directory, bool ignore_fail = false);
	bool remove(const string& file);
	FILE* fopen_mkdirs(const string& path, const string& mode);
	void setRandomSeed(int seed);

	//[low, high)
	float randrf(float low, float high);
	cv::Rect randbox(cv::Size size, cv::Size limit);

	//[low, high)
	int randr(int low, int high);
	int randr(int high);
	int randr_exclude(int mi, int mx, int exclude);

	//[low, end)
	vector<int> seque(int begin, int end);
	vector<int> seque(int end);
	vector<int> shuffleSeque(int begin, int end);
	vector<int> shuffleSeque(int end);

	template<typename _T>
	_T& randitem(vector<_T>& arr) {
		int n = randr(0, (int)arr.size());
		return arr[n];
	}

	template<typename _T>
	const _T& randitem(const vector<_T>& arr) {
		int n = randr(0, (int)arr.size());
		return arr[n];
	}

	template<typename _T>
	void repeat(vector<_T>& list, int count, bool requirement_count_matched = false) {

		int oldsize = list.size();
		int n = count - list.size();
		for (int i = 0; i < n; ++i)
			list.push_back(list[i % oldsize]);

		if (requirement_count_matched) {
			if (list.size() > count) {
				shuffle(list);
				list.erase(list.begin() + count, list.end());
			}
		}
	}

	vector<cv::Scalar> randColors(int size);
	cv::Scalar randColor(int label, int size = 80);
	const vector<string>& vocLabels();
	const vector<string>& cocoLabels();
	int vocLabel(const string& name);
	int cocoLabel(const string& name);
	string cocoName(int label);

	enum DrawType : int {
		CoCo = 0,
		Voc = 1,
		Custom = 2,
		NoName = 3,
		Empty = 4
	};
	void drawbbox(cv::Mat& image, const BBox& bbox, DrawType drawType = DrawType::CoCo, const string& labelText = "");

	template<typename _TArray>
	inline _TArray& appendArray(_TArray& array, _TArray& other) {
		array.insert(array.end(), other.begin(), other.end());
		return array;
	}

	template<typename _T>
	float l2distance(const _T& a, const _T& b) {
		return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
	}

	//32 uuid
	string uuid();

	class BinIO {
	public:
		enum Head {
			FileIO = 0,
			MemoryRead = 1,
			MemoryWrite = 2
		};

		BinIO() { openMemoryWrite(); }
		BinIO(const void* ptr, int memoryLength = -1) { openMemoryRead(ptr, memoryLength); }
		BinIO(const string& file, const string& mode, bool mkparents = false);
		BinIO(FILE* fhandle);
		virtual ~BinIO();
		bool opened();
		bool openFile(const string& file, const string& mode, bool mkparents = false);
		void openFile(FILE* file);
		bool openMemoryRead(const void* ptr, int memoryLength = -1);
		void openMemoryWrite();
		const string& writedMemory() { return memoryWrite_; }
		void close();
		int write(const void* pdata, size_t length);
		int writeData(const string& data);
		int read(void* pdata, size_t length);
		string readData(int numBytes);
		int readInt();
		float readFloat();
		bool eof();

		BinIO& operator >> (string& value);
		BinIO& operator << (const string& value);
		BinIO& operator << (const char* value);
		BinIO& operator << (const vector<string>& value);
		BinIO& operator >> (vector<string>& value);

		BinIO& operator >> (cv::Mat& value);
		BinIO& operator << (const cv::Mat& value);

		template<typename _T>
		BinIO& operator >> (vector<_T>& value) {
			int length = 0;
			(*this) >> length;

			value.resize(length);
			read(value.data(), length * sizeof(_T));
			return *this;
		}

		template<typename _T>
		BinIO& operator << (const vector<_T>& value) {
			(*this) << (int)value.size();
			write(value.data(), sizeof(_T) * value.size());
			return *this;
		}

		template<typename _T>
		BinIO& operator >> (_T& value) {
			read(&value, sizeof(_T));
			return *this;
		}

		template<typename _T>
		BinIO& operator << (const _T& value) {
			write(&value, sizeof(_T));
			return *this;
		}

		bool opstate() const {
			return opstate_;
		}

	private:
		size_t readModeEndSEEK_ = 0;
		FILE* f_ = nullptr;
		string memoryWrite_;
		const char* memoryRead_ = nullptr;
		int memoryCursor_ = 0;
		int memoryLength_ = -1;
		Head flag_ = MemoryWrite;
		bool opstate_ = true;
	};

	///////////////////////////////////////////////////////////////////////
	class FileCache {

	public:
		FileCache(int maxCacheSize = -1);
		string loadfile(const string& file);
		vector<LabBBox> loadxml(const string& file, int* width, int* height, const string& filter);
		cv::Mat loadimage(const string& file, int color = 1);

	private:
		bool hitFile(const string& file);

	private:
		std::mutex lock_;
		int maxCacheSize_ = 0;
		map<string, string> hits_;
		vector<string> cacheNames_;
	};


	template<typename _Input>
	class ThreadPoolOnlyInput {
	public:

		typedef _Input Input;
		typedef _Input Job;

		virtual ~ThreadPoolOnlyInput() {
			stop();
		}

		void stop() {

			if (!worker_run_)
				return;

			worker_run_ = false;
			cv_.notify_all();

			for (auto& t : ts_) {
				if (t.joinable())
					t.join();
			}
			ts_.clear();
		}

		virtual bool start(int num_threads = 1) {

			stop();
			worker_run_ = true;

			std::vector<std::promise<bool>> pro(num_threads);
			for (int i = 0; i < num_threads; ++i)
				ts_.emplace_back(std::bind(&ThreadPoolOnlyInput::worker, this, i, std::ref(pro[i])));

			bool finished = true;
			for (int i = 0; i < num_threads; ++i) 
				finished = finished && pro[i].get_future().get();

			if (!finished) 
				stop();

			return finished;
		}

		virtual void commit(const _Input& data) {
			{
				std::unique_lock<std::mutex> l(job_lock_);
				jobs_.push(data);
			};
			cv_.notify_one();
		}

	protected:
		bool get_job_and_wait(Job& job) {
			std::unique_lock<std::mutex> l(job_lock_);
			cv_.wait(l, [&] {return !worker_run_ || !jobs_.empty(); });

			if (!worker_run_)
				return false;

			job = jobs_.front();
			jobs_.pop();
			return true;
		}

		bool get_jobs_and_wait(std::vector<Job>& jobs_result, int max_num = 1) {
			std::unique_lock<std::mutex> l(job_lock_);
			cv_.wait(l, [&] {return !worker_run_ || !jobs_.empty(); });

			if (!worker_run_)
				return false;

			jobs_result.clear();
			int n = std::min(max_num, (int)jobs_.size());
			for (int i = 0; i < n; ++i) {
				jobs_result.emplace_back(jobs_.front());
				jobs_.pop();
			}
			return true;
		}

		virtual void worker(int tid, std::promise<bool>& pro) = 0;

	protected:
		volatile bool worker_run_ = false;

	protected:
		vector<std::thread> ts_;
		std::condition_variable cv_;
		std::mutex job_lock_;
		std::queue<Job> jobs_;
	};


	template<typename _Input, typename _Output>
	class ThreadPool {
	public:

		typedef _Input Input;
		typedef _Output Output;

		struct Job {
			_Input data;
			std::shared_ptr<std::promise<_Output>> pro;

			Job() {}
			Job(const _Input& data):data(data) {}
		};

		virtual ~ThreadPool() {
			stop();
		}

		void stop() {

			if (!worker_run_)
				return;

			worker_run_ = false;
			cv_.notify_all();

			for (auto& t : ts_) {
				if (t.joinable())
					t.join();
			}
			ts_.clear();
		}

		bool start(int num_threads = 1) {

			stop();
			worker_run_ = true;

			std::vector<std::promise<bool>> pro(num_threads);
			for (int i = 0; i < num_threads; ++i)
				ts_.emplace_back(std::bind(&ThreadPool::worker, this, i, std::ref(pro[i])));

			bool finished = true;
			for (int i = 0; i < num_threads; ++i) 
				finished = finished && pro[i].get_future().get();

			if (!finished) 
				stop();

			return finished;
		}

		std::shared_future<_Output> commit(const _Input& data) {

			Job job(data);
			job.pro.reset(new std::promise<_Output>());
			{
				std::unique_lock<std::mutex> l(job_lock_);
				jobs_.push(job);
			};
			cv_.notify_one();
			return job.pro->get_future();
		}

	protected:
		bool get_job_and_wait(Job& job) {
			std::unique_lock<std::mutex> l(job_lock_);
			cv_.wait(l, [&] {return !worker_run_ || !jobs_.empty(); });

			if (!worker_run_)
				return false;

			job = jobs_.front();
			jobs_.pop();
			return true;
		}

		bool get_jobs_and_wait(std::vector<Job>& jobs_result, int max_num = 1) {
			std::unique_lock<std::mutex> l(job_lock_);
			cv_.wait(l, [&] {return !worker_run_ || !jobs_.empty(); });

			if (!worker_run_)
				return false;

			jobs_result.clear();
			int n = std::min(max_num, (int)jobs_.size());
			for (int i = 0; i < n; ++i) {
				jobs_result.emplace_back(jobs_.front());
				jobs_.pop();
			}
			return true;
		}

		virtual void worker(int tid, std::promise<bool>& pro) = 0;

	protected:
		volatile bool worker_run_ = false;

	protected:
		vector<std::thread> ts_;
		std::condition_variable cv_;
		std::mutex job_lock_;
		std::queue<Job> jobs_;
	};

	void setThreadContext(void* ptr);
	void* getThreadContext();
};

#endif //CC_UTILS_HPP