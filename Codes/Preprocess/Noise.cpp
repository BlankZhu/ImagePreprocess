#include "stdafx.h"
#include "Noise.h"

#include <random>
#include <ctime>

// definitions for class Noise
Noise::Noise()
{
}

Noise::~Noise()
{
}


// definitions for class GaussianNoise
GaussianNoise::GaussianNoise(double mean, double stddev)
	:  mean_(mean), stddev_(stddev) { }

GaussianNoise::~GaussianNoise()
{
}

void
GaussianNoise::ApplyNoise(cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(!src.empty());

	cv::Mat src_16SC;
	cv::Mat gaussian_noise = cv::Mat(src.size(), CV_16SC3);
	cv::randn(gaussian_noise, cv::Scalar::all(mean_), cv::Scalar::all(stddev_));

	src.convertTo(src_16SC, CV_16SC3);
	cv::addWeighted(src_16SC, 1.0, gaussian_noise, 1.0, 0.0, src_16SC);

	src_16SC.convertTo(dst, src.type());
}

// definitions for class SaltNoise
SaltNoise::SaltNoise(double probability)
	: probability_(probability) { }

SaltNoise::~SaltNoise()
{
}

void
SaltNoise::ApplyNoise(cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(!src.empty());

	cv::Mat tmp = src.clone();

	// engine for random number
	std::default_random_engine engine(time(nullptr));
	std::uniform_real_distribution<> dis(0.0, 1.0);

	// travers the pixels
	if (tmp.channels() == 1)
	{
		auto it = tmp.begin<uchar>();
		auto it_end = tmp.end<uchar>();
		while (it != it_end)
		{
			auto rate = dis(engine);
			if (rate <= probability_)
			{
				// salt noise, apply white pixels
				(*it) = 255;
			}
			++it;
		}
	}
	if (tmp.channels() == 3)
	{
		auto it = tmp.begin<cv::Vec3b>();
		auto it_end = tmp.end<cv::Vec3b>();
		while (it != it_end)
		{
			auto rate = dis(engine);
			if (rate <= probability_)
			{
				// salt noise, apply white pixels
				(*it)[0] = 255;
				(*it)[1] = 255;
				(*it)[2] = 255;
			}
			++it;
		}
	}

	tmp.copyTo(dst);
}


// definitions for class PepperNoise
PepperNoise::PepperNoise(double probability)
	: probability_(probability) { }

PepperNoise::~PepperNoise()
{
}

void
PepperNoise::ApplyNoise(cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(!src.empty());

	cv::Mat tmp = src.clone();

	// engine for random number
	std::default_random_engine engine(time(nullptr));
	// std::uniform_real_distribution<> dis(0.0, 1.0);
	std::bernoulli_distribution dis(probability_);

	// travers the pixels
	if (tmp.channels() == 1)
	{
		auto it = tmp.begin<uchar>();
		auto it_end = tmp.end<uchar>();
		while (it != it_end)
		{
			auto rate = dis(engine);
			if (rate == true)
			{
				// salt noise, apply black pixels
				(*it) = 0;
			}
			++it;
		}
	}
	if (tmp.channels() == 3)
	{
		auto it = tmp.begin<cv::Vec3b>();
		auto it_end = tmp.end<cv::Vec3b>();
		while (it != it_end)
		{
			auto rate = dis(engine);
			if (rate == true)
			{
				// salt noise, apply black pixels
				(*it)[0] = 0;
				(*it)[1] = 0;
				(*it)[2] = 0;
			}
			++it;
		}
	}

	tmp.copyTo(dst);
}


// definitions for class PeriodicNoise
PeriodicNoise::PeriodicNoise(bool vertical, double period, uchar intensity)
	: vertical_(vertical), period_(period), intensity_(intensity) { }

PeriodicNoise::~PeriodicNoise()
{
}

void 
PeriodicNoise::ApplyNoise(cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(!src.empty());

	// generate noise vertically
	if (vertical_ == true)
	{
		cv::Mat tmp;
		cv::Mat periodic_noise;
		if (src.channels() == 1)
		{
			periodic_noise = cv::Mat(src.size(), CV_16UC1);
			src.convertTo(tmp, CV_16UC1);

			// generate the periodic noise, vetically
			int width_limit = periodic_noise.channels() * periodic_noise.cols;
			for (size_t height = 0; height < periodic_noise.rows; ++height)
			{
				uchar *data = periodic_noise.ptr<uchar>(height);
				uchar noise_intensity = static_cast<uchar>((sin(height * 2.0 * CV_PI / period_) + 1.0) / 2.0 * intensity_);
				for (size_t width = 0; width < width_limit; ++width)
					data[width] = noise_intensity;
			}
		}
		if (src.channels() == 3)
		{
			periodic_noise = cv::Mat(src.size(), CV_16UC3);
			src.convertTo(tmp, CV_16UC3);

			// generate the periodic noise, vetically
			int width_limit = periodic_noise.channels() * periodic_noise.cols;
			for (size_t height = 0; height < periodic_noise.rows; ++height)
			{
				uchar *data = periodic_noise.ptr<uchar>(height);
				uchar noise_intensity = static_cast<uchar>((sin(height * 2.0 * CV_PI / period_) + 1.0) / 2.0 * intensity_);
				for (size_t width = 0; width < width_limit * 2; ++width)	// because we use CV_16UC3, so we mul 2 here
					data[width] = noise_intensity;
			}
		}
		// we set the first weight to 0.99 because the noise will brighten the raw image
		//.. so we just darken the raw image a little bit
		cv::addWeighted(tmp, 0.99, periodic_noise, 0.01, 0.0, tmp);
		tmp.convertTo(dst, src.type());
	}
	// generate noise horizontally
	else
	{
		cv::Mat tmp;
		cv::Mat periodic_noise;
		if (src.channels() == 1)
		{
			periodic_noise = cv::Mat(src.size(), CV_8UC1);

			// generate the periodic noise, horizontally
			int width_limit = periodic_noise.channels() * periodic_noise.cols;
			for (size_t height = 0; height < periodic_noise.rows; ++height)
			{
				uchar *data = periodic_noise.ptr<uchar>(height);
				for (size_t width = 0; width < width_limit * 2; )
				{
					uchar noise_intensity = static_cast<uchar>((sin(width * 2.0 * CV_PI / period_) + 1.0) / 2.0 * intensity_);
					data[width] = noise_intensity;

					width += 1;	// to the next pixel
				}
			}
		}
		if (src.channels() == 3)
		{
			periodic_noise = cv::Mat(src.size(), CV_16UC3);
			src.convertTo(tmp, CV_16UC3);

			// generate the periodic noise, vetically
			int width_limit = periodic_noise.channels() * periodic_noise.cols;
			for (size_t height = 0; height < periodic_noise.rows; ++height)
			{
				uchar *data = periodic_noise.ptr<uchar>(height);
				for (size_t width = 0; width < width_limit * 2;)
				{
					// width * 3 because we have three channels to do the addition
					uchar noise_intensity = static_cast<uchar>((sin(width * 3 * 2.0 * CV_PI / period_) + 1.0) / 2.0 * intensity_);
					// aligned to the higher address, we use data[width+1] but not data[width] 
					data[width + 1] = noise_intensity;
					data[width + 3] = noise_intensity;
					data[width + 5] = noise_intensity;
					width += 6;
				}
			}
		}
		cv::addWeighted(tmp, 0.99, periodic_noise, 0.01, 0.0, tmp);
		tmp.convertTo(dst, src.type());
	}
}