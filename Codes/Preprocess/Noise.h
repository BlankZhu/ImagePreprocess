#pragma once

#include <opencv2/core.hpp>

/*
	CLASS: Noise
	Abstract Class, base class for all noise.
*/
class Noise
{
public:
	Noise();
	virtual ~Noise();

	/* \brief
		Noise function to be overrided.
		It will apply noise to the src Mat.
		\param[in] cv::Mat &src source Mat
		\param[in] cv::Mat &dst destination Mat
	*/
	virtual void ApplyNoise(cv::Mat& src, cv::Mat& dst) = 0;
};



/*
	CLASS: GaussianNoise
	Implemented Noise, apply gaussian noise.
*/
class GaussianNoise : public Noise
{
public:
	/* \brief
		Default constructor of gaussian noise.
		It will apply mean = 1.0 and stddev = 1.0 by default.
		\param[in] double mean mean of the gaussian
		\param[in] double stddev standard deviation of the gaussian
	*/
	GaussianNoise(double mean = 1.0, double stddev = 10.0);

	/* \brief
		Destructor of gaussian noise
	*/
	virtual ~GaussianNoise();

	/* \brief
		Noise function to be overrided.
		It will apply noise to the src Mat.
		\param[in] cv::Mat &src source Mat
		\param[in] cv::Mat &dst destination Mat
	*/
	void ApplyNoise(cv::Mat& src, cv::Mat& dst);

private:
	double mean_;
	double stddev_;
};



/*
	CLASS: SaltNoise
	Implemented Noise, apply salty noise.
*/
class SaltNoise : public Noise
{
public:
	/* \brief
		Default constructor of salt noise
		It will apply probability = 0.05 by default
		\param[in] double probability probability of the salt noise
	*/
	SaltNoise(double probability = 0.05);

	/* \brief
		Destructor of salt noise
	*/
	virtual ~SaltNoise();

	/* \brief
		Noise function overrided.
		It will apply noise to the src Mat.
		\param[in] cv::Mat &src source Mat
		\param[in] cv::Mat &dst destination Mat
	*/
	virtual void ApplyNoise(cv::Mat& src, cv::Mat& dst);

private:
	double probability_;
};



/*
	CLASS: PepperNoise
	Implemented Noise, apply pepper noise.
*/
class PepperNoise : public Noise
{
public:
	/* \brief
		Default constructor of pepper noise
		It will apply probability = 0.05 by default
		\param[in] double probability probability of the salt noise
	*/
	PepperNoise(double probability = 0.05);

	/* \brief
		Destructor of pepper noise
	*/
	virtual ~PepperNoise();

	/* \brief
		Noise function overrided.
		It will apply noise to the src Mat.
		\param[in] cv::Mat &src source Mat
		\param[in] cv::Mat &dst destination Mat
	*/
	virtual void ApplyNoise(cv::Mat& src, cv::Mat& dst);

private:
	double probability_;
};



/*
	CLASS: PeriodicNoise
	Implemented Noise, apply periodic noise.
*/
class PeriodicNoise : public Noise
{
public:
	/* \brief
		Default constructor of periodic noise
		It is based on sin, and will apply noise vertically by default.
		\param[in] bool vertical vertical if true, otherwise, horizontal
		\param[in] double period period of the sin function
		\param[in] uchar intensity of the noise, SUGGEST to be smaller then 64
	*/
	PeriodicNoise(bool vertical = true, double period = 2 * CV_PI, uchar intensity = 8);

	/* \brief
		Destructor of periodic noise
	*/
	virtual ~PeriodicNoise();

	/* \brief
		Noise function overrided.
		It will apply noise to the src Mat.
		\param[in] cv::Mat &src source Mat
		\param[in] cv::Mat &dst destination Mat
	*/
	virtual void ApplyNoise(cv::Mat& src, cv::Mat& dst);

private:
	bool vertical_;
	double period_;
	uchar intensity_;
};


/*
	Add your own noise in this form:
	class YourNoise : public Noise
	{
		public:
		// 1. your constructors & destructors
		// 2. your member functions
		// 3. override ApplyNoise function
		private:
		// 1. your data members
		// 2. your tool functions
	}

*/