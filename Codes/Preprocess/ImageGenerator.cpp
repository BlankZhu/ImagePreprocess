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
#include <list>

ImageGenerator::ImageGenerator() :
	hist_eqlize_(true), mean_normalization_(true),
	rotate_(true), rtt_min_(0), rtt_max_(360.0),
	v_move_(true), v_min_(-50), v_max_(50),
	h_move_(true), h_min_(-50), h_max_(50),
	exchange_chan_(true),
	filling_method_(cv::BORDER_CONSTANT), cval_(0),
	noises_(std::vector<std::function<void(cv::Mat&)>>()),
	resize_(true), rsz_min_(0.5), rsz_max_(1.5)
{
}

ImageGenerator::ImageGenerator(bool hist_equlize, bool mean_normalization,
	bool rotate, double rotate_min, double rotate_max,
	bool v_move, int v_move_min, int v_move_max, 
	bool h_move, int h_move_min, int h_move_max,
	bool exchange_chan, int filling_method, int cval,
	bool v_flip, bool h_flip,
	std::vector<std::function<void(cv::Mat&)>> noises,
	bool resize, double rsz_min, double rsz_max) : 
	hist_eqlize_(hist_equlize), mean_normalization_(mean_normalization),
	rotate_(rotate), rtt_min_(rotate_min), rtt_max_(rotate_max),
	v_move_(v_move), v_min_(v_move_min), v_max_(v_move_max),
	h_move_(h_move), h_min_(h_move_min), h_max_(h_move_max),
	exchange_chan_(exchange_chan), filling_method_(filling_method_),
	cval_(cval), v_flip_(v_flip), h_flip_(h_flip),
	noises_(noises),
	resize_(resize), rsz_min_(rsz_min), rsz_max_(rsz_max)
{
}

ImageGenerator::~ImageGenerator()
{
}

void 
ImageGenerator::HistEqualize(cv::Mat &src, std::vector<cv::Mat> &res)
{
	CV_Assert(!src.empty());

	cv::Mat tmp;
	src.copyTo(tmp);

	if (src.channels() == 1)
	{
		cv::equalizeHist(src, tmp);
		res.push_back(tmp.clone());
	}
	if (src.channels() == 3)
	{
		std::vector<cv::Mat> channels;

		// split
		cv::split(tmp, channels);

		// equalize
		for (auto &m : channels)
			cv::equalizeHist(m, m);

		// merge
		cv::merge(channels, tmp);

		res.push_back(tmp.clone());
	}
	return;
}

void 
ImageGenerator::MeanNormalize(cv::Mat &src, std::vector<cv::Mat> &res)
{
	CV_Assert(!src.empty());

	// to store the result image
	cv::Mat tmp;
	src.copyTo(tmp);

	// traverse grayscale image
	if (src.channels() == 1)
	{
		cv::Mat_<uchar>::iterator it = tmp.begin<uchar>();
		cv::Mat_<uchar>::iterator it_end = tmp.end<uchar>();

		// get mean
		auto mean = cv::mean(tmp);

		// substraction, mind zero value
		for (; it != it_end; ++it)
		{
			if (mean[0] > (*it))
				(*it) = 0;
			else
				(*it) -= mean[0];
		}
	}
	// traverse BGR image
	if (src.channels() == 3)
	{
		cv::Mat_<cv::Vec3b>::iterator it = tmp.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::iterator it_end = tmp.end<cv::Vec3b>();

		// calc mean for rgb
		auto means = cv::mean(tmp);

		// substraction, mind 0
		for (; it != it_end; ++it)
		{
			if ((*it)[0] < means[0])
				(*it)[0] = 0;
			else
				(*it)[0] -= means[0];

			if ((*it)[1] < means[1])
				(*it)[1] = 0;
			else
				(*it)[1] -= means[1];

			if ((*it)[2] < means[2])
				(*it)[2] = 0;
			else
				(*it)[2] -= means[2];
		}
	}

	res.push_back(tmp.clone());

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
	res.push_back(tmp.clone());
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

	// set triangle points for affine translation
	cv::Point2f src_tri[3];
	cv::Point2f dst_tri[3];

	src_tri[0] = cv::Point2f(0, 0);
	src_tri[1] = cv::Point2f(src.cols - 1, 0);
	src_tri[2] = cv::Point2f(0, src.rows - 1);

	dst_tri[0] = cv::Point2f(0, step);
	dst_tri[1] = cv::Point2f(src.cols - 1, step);
	dst_tri[2] = cv::Point2f(0, src.rows - 1 + step);

	// get affine matrix
	auto warp_mat = cv::getAffineTransform(src_tri, dst_tri);
	// do affine transform
	cv::warpAffine(tmp, tmp, warp_mat, tmp.size());

	res.push_back(tmp.clone());
}

void
ImageGenerator::HMove(cv::Mat & src, std::vector<cv::Mat>& res,
	int h_min, int h_max)
{
	CV_Assert(!src.empty());

	if (h_min > h_max)
		return;
	if (h_min == h_max)
	{
		h_min = static_cast<int> (-(src.cols / 2));
		h_max = -h_min;
	}

	// set random steps
	std::default_random_engine engine(time(nullptr));
	std::uniform_int_distribution<> dis(h_min, h_max);
	auto step = dis(engine);

	cv::Mat tmp;
	if (step < 0)
		cv::copyMakeBorder(src, tmp, 0, 0, fabs(step), 0, filling_method_, cval_);
	else
		cv::copyMakeBorder(src, tmp, 0, 0, 0, step, filling_method_, cval_);

	// set triangle points for affine translation
	cv::Point2f src_tri[3];
	cv::Point2f dst_tri[3];

	src_tri[0] = cv::Point2f(0, 0);
	src_tri[1] = cv::Point2f(src.cols - 1, 0);
	src_tri[2] = cv::Point2f(0, src.rows - 1);

	dst_tri[0] = cv::Point2f(step, 0);
	dst_tri[1] = cv::Point2f(src.cols - 1 + step, 0);
	dst_tri[2] = cv::Point2f(step, src.rows - 1);

	// get affine matrix
	auto warp_mat = cv::getAffineTransform(src_tri, dst_tri);
	// do affine transform
	cv::warpAffine(tmp, tmp, warp_mat, tmp.size());

	res.push_back(tmp.clone());
}

void 
ImageGenerator::ChangeChannel(cv::Mat & src, std::vector<cv::Mat>& res)
{
	CV_Assert(!src.empty());
	cv::Mat tmp_src = src.clone();

	if (tmp_src.channels() == 1)
	{
		// Why do you exhange the channel of a grey scale image?
		return;
	}
	if (tmp_src.channels() == 3)
	{
		uchar tmp_ch_val = 0;


		// OK, let's do some dirty and stupid work
		cv::Mat tmp1;
		ShiftChannel(src, tmp1);
		res.push_back(tmp1.clone());

		cv::Mat tmp2;
		ShiftChannel(tmp1, tmp2);
		res.push_back(tmp2.clone());

		// RGB -> RBG, we now change a sequence
		cv::Mat tmp3;
		tmp_src.copyTo(tmp3);
		auto it = tmp3.begin<cv::Vec3b>();
		auto it_end = tmp3.end<cv::Vec3b>();
		for (; it != it_end; ++it)
		{
			tmp_ch_val = (*it)[1];
			(*it)[1] = (*it)[2];
			(*it)[2] = tmp_ch_val;
		}
		res.push_back(tmp3.clone());

		// continue our dirty work
		cv::Mat tmp4;
		ShiftChannel(tmp3, tmp4);
		res.push_back(tmp4.clone());

		cv::Mat tmp5;
		ShiftChannel(tmp4, tmp5);
		res.push_back(tmp5.clone());
	}

	return;
}

void 
ImageGenerator::ShiftChannel(cv::Mat & src, cv::Mat & dst)
{
	CV_Assert(!src.empty());
	src.copyTo(dst);

	uchar tmp_ch_val = 0;
	auto it = dst.begin<cv::Vec3b>();
	auto it_end = dst.end<cv::Vec3b>();
	for (; it != it_end; ++it)
	{
		tmp_ch_val = (*it)[0];
		(*it)[0] = (*it)[1];
		(*it)[1] = (*it)[2];
		(*it)[2] = tmp_ch_val;
	}
	return;
}

void 
ImageGenerator::VFlip(cv::Mat & src, std::vector<cv::Mat>& res)
{
	CV_Assert(!src.empty());

	cv::Mat tmp;

	// flip, vertically
	cv::flip(src, tmp, 0);

	// push to res
	res.push_back(tmp.clone());
}

void
ImageGenerator::HFlip(cv::Mat & src, std::vector<cv::Mat>& res)
{
	CV_Assert(!src.empty());

	cv::Mat tmp;

	// flip, horizontally
	cv::flip(src, tmp, 1);

	// push to res
	res.push_back(tmp.clone());
}

void 
ImageGenerator::ApplyNoise(cv::Mat & src, 
	std::vector<cv::Mat>& res, 
	std::vector<std::function<void(cv::Mat&)>>& noises)
{
	// apply nosies
	for (size_t cnt = 0; cnt < noises.size(); ++cnt)
	{
		cv::Mat tmp;
		src.copyTo(tmp);
		(noises[cnt])(tmp);
		res.push_back(tmp.clone());
	}
}

void 
ImageGenerator::Resize(cv::Mat & src, std::vector<cv::Mat>& res, 
	double rsz_min, double rsz_max)
{
	CV_Assert(!src.empty());
	
	// params check
	if (rsz_min > rsz_max)
		return;
	if (rsz_min < 0)
		rsz_min = 0;
	if (rsz_max < rsz_min)
		rsz_max = 1;

	// set random resize rate
	std::default_random_engine engine(time(nullptr));
	std::uniform_real_distribution<> dis(rsz_min, rsz_max);
	auto resize_rate = dis(engine);

	// do resize
	cv::Mat tmp;
	cv::resize(src, tmp,
		cv::Size(static_cast<size_t>(src.cols * resize_rate),
			static_cast<size_t>(src.rows * resize_rate)));
	
	res.push_back(tmp.clone());
}

void
ImageGenerator::gen(cv::Mat & src, std::vector<cv::Mat>& res)
{
	CV_Assert(!src.empty());
	res.clear();
	res.push_back(src.clone());

	// val to count the size of res
	int tmp_sz = res.size();

	// start generating
	// HistEqulize
	if (hist_eqlize_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			HistEqualize(res[idx], res);
	}
	// MeanNormalize
	if (mean_normalization_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			MeanNormalize(res[idx], res);
	}
	// Rotate
	if (rotate_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			Rotate(res[idx], res, rtt_min_, rtt_max_);
	}
	// vertical move
	if (v_move_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			VMove(res[idx], res, v_min_, v_max_);
	}
	// horizontal move
	if (h_move_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			VMove(res[idx], res, h_min_, h_max_);
	}
	// change channel
	if (exchange_chan_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			ChangeChannel(res[idx], res);
	}
	// vertical flip
	if (v_flip_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			VFlip(res[idx], res);
	}
	// vertical flip
	if (h_flip_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			HFlip(res[idx], res);
	}
	// apply noise
	if (!noises_.empty())
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			ApplyNoise(res[idx], res, noises_);
	}
	// resize
	if (resize_ == true)
	{
		tmp_sz = res.size();
		for (size_t idx = 0; idx < tmp_sz; ++idx)
			Resize(res[idx], res, rsz_min_, rsz_max_);
	}

	return;
}

void ImageGenerator::debug_gen(cv::Mat & src, std::vector<cv::Mat>& res)
{
	// DEBUGGING
	// HistEqualize(src, res);   TEST GOOD!
	// MeanNormalize(src, res);  TEST GOOD!
	// Rotate(src, res, rtt_min_, rtt_max_);	TEST GOOD!
	// VMove(src, res, -100, 0);				TEST OK, Check if you need a fix size of image
	// HMove(src, res, 10, 100);				TEST OK, Check if you need a fix size of image
	// ChangeChannel(src, res);	 TEST GOOD!
	// VFlip(src, res);			 TEST GOOD!
	// HFlip(src, res);			 TEST GOOD!
	// ApplyNoise(src, noises);	 TODO!!!
	// Resize(src, res, 0.5, 2.0);	TEST GOOD!
}