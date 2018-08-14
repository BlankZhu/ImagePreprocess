#include "stdafx.h"
#include "ImageGenerator.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <random>
#include <ctime>

ImageGenerator::ImageGenerator() :
	hist_eqlize_(true), ZCA_whiten_(true),
	rtt_min_(0), rtt_max_(360.0),
	v_min_(0), v_max_(0),
	h_min_(0), h_max_(0),
	exchange_chan_(true),
	filling_method_(cv::BORDER_CONSTANT), cval_(0),
	noises_(std::vector<std::function<void(cv::Mat&)>>()),
	rsz_min_(1), rsz_max_(1)
{
}

ImageGenerator::ImageGenerator(bool hist_equlize, bool ZCA_whiten,
	double rotate_min, double rotate_max,
	int v_move_min, int v_move_max, int h_move_min, int h_move_max,
	bool exchange_chan, int filling_method, int cval,
	bool v_flip, bool h_flip,
	std::vector<std::function<void(cv::Mat&)>> noises,
	double rsz_min, double rsz_max) : 
	hist_eqlize_(hist_equlize), ZCA_whiten_(ZCA_whiten),
	rtt_min_(rotate_min), rtt_max_(rotate_max),
	v_min_(v_move_min), v_max_(v_move_max),
	h_min_(h_move_min), h_max_(h_move_max),
	exchange_chan_(exchange_chan), filling_method_(filling_method_),
	cval_(cval), v_flip_(v_flip), h_flip_(h_flip),
	noises_(noises), rsz_min_(rsz_min), rsz_max_(rsz_max)
{
}

ImageGenerator::~ImageGenerator()
{
}

void 
ImageGenerator::HistEqualize(cv::Mat &src, std::vector<cv::Mat> &res)
{
	CV_Assert(!src.empty());

	cv::Mat dst;
	cv::equalizeHist(src, dst);
	res.push_back(dst);
}

// TODO!
void 
ImageGenerator::ZCAWhiten(cv::Mat &src, std::vector<cv::Mat> &res)
{
	CV_Assert(!src.empty());

	// TODO
	return;
}

void 
ImageGenerator::Rotate(cv::Mat &src, std::vector<cv::Mat> &res, 
	double r_min, double r_max)
{
	CV_Assert(!src.empty());

	// parameter check
	if (r_min > r_max)
		return;
	if (r_min < 0)
		r_min = 0;
	if (r_max > 360.0)
		r_max = 360.0;
	if (r_min == r_max)
	{
		r_min = 0;
		r_max = 360.0;
	}

	// set Mat
	cv::Mat tmp;
	
	// set random degree
	std::default_random_engine engine(time(nullptr));
	std::uniform_real_distribution<> dis(r_min, r_max);
	auto rtt_angle = dis(engine);
	auto radian = static_cast<float>(rtt_angle / 180.0 * CV_PI);

	// pad the image for rotation
	auto max = (src.cols > src.rows ? src.cols : src.rows);
	int uniSize = static_cast<int>(max * 1.414);
	int dx = static_cast<int>((uniSize - src.cols) / 2);
	int dy = static_cast<int>((uniSize - src.rows) / 2);
	cv::copyMakeBorder(src, tmp, dy, dy, dx, dx, filling_method_, cval_);

	// get rotation center
	cv::Point2f center(static_cast<float>(tmp.cols / 2), static_cast<float>(tmp.rows / 2));

	// get rotation matrix
	auto rotation_matrix = cv::getRotationMatrix2D(center, rtt_angle, 1);

	// do affine transform
	cv::warpAffine(tmp, tmp, rotation_matrix, tmp.size());

	// size of rotated image
	float sin_val = fabs(sin(radian));
	float cos_val = fabs(cos(radian));
	cv::Size target_sz(static_cast<int>(src.cols * cos_val + src.rows * sin_val),
		static_cast<int>(src.cols * sin_val + src.rows * cos_val));

	// cut away the surroundings 
	int x = (tmp.cols - target_sz.width) / 2;
	int y = (tmp.rows - target_sz.height) / 2;
	cv::Rect rect(x, y, target_sz.width, target_sz.height);
	tmp = cv::Mat(tmp, rect);

	// push to res
	res.push_back(tmp);
}

void 
ImageGenerator::VMove(cv::Mat & src, std::vector<cv::Mat>& res, 
	int v_min, int v_max)
{
	CV_Assert(!src.empty());

	if (v_min > v_max)
		return;
	if (v_min == v_max)
	{
		v_min = static_cast<int> (-(src.rows / 2));
		v_max = -v_min;
	}

	// set random steps
	std::default_random_engine engine(time(nullptr));
	std::uniform_int_distribution<> dis(v_min, v_max);
	auto step = dis(engine);

	cv::Mat tmp;
	if (step < 0)
		cv::copyMakeBorder(src, tmp, 0, fabs(step), 0, 0, filling_method_, cval_);
	else
		cv::copyMakeBorder(src, tmp, step, 0, 0, 0, filling_method_, cval_);

	// TODO: fill border move or mirror like move?
}