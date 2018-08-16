#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <cmath>

#include <opencv2/core.hpp>

#include "Noise.h"

class ImageGenerator
{
public:
	/*
	 * Constructor
	 */
	// default constructor
	/* \brief
		This constructor initialize a default ImageGenerator.
		It WILL apply Histogram Equalization.
		It WILL apply Mean Normalization.
		It WILL randomly rotate the image in [0, 360.0).
		It WILL vertically move the image in [-50, 50).
		It WILL horizontally move the image in [-50, 50).
		It WILL exchange the RGB channels.
		It WILL use zero padding as default filling method.
		It WILL NOT add any noise on the image
		It WILL resize the image in [0.5, 1.5)
	*/
	ImageGenerator();

	// other constructor
	/* \brief
		This constructor initialize a fixed ImageGenerator.
		It WILL apply Histogram Equalization according to hist_eqlize.
		It WILL apply Mean Normalization according to mean_normalization.
		It WILL randomly rotate the image in [rotate_min, rotate_max).
			..if you dont want to rotate, set rotate == false
		It WILL vertically move the image in [v_move_min, v_move_max).
			..if you dont want to move, set v_move == false
		It WILL horizontally move the image in [h_move_min, h_move_max).
			..if you dont want to move, set h_move == false
		It may exchange the channels according to exchange_chan.
		It WILL use filling method specifed by filling_method.
			..if you use constant filling, you should specify cval,
			..otherwise the cval is useless, you can set a lucky number you like.
		It WILL add noises to image.
			..if you dont want to add noise, use an empty vector
		It WILL resize the image in [rsz_min, rsz_max).
			..if you dont want to resize, set resize == false
	 * \param[in] bool hist_equlize use or not
	 * \param[in] bool mean_normalization use or not
	 * \param[in] bool rotate use or not
	 * \param[in] double rotate_min minimun degree
	 * \param[in] double rotate_max maximun degree
	 * \param[in] bool v_move use vertical move or not
	 * \param[in] int v_move_min minimun pixels
	 * \param[in] int v_move_max maximun pixels
	 * \param[in] bool h_move use horizontal move or not
	 * \param[in] int h_move_min mimimun pixels
	 * \param[in] int h_move_max maximun pixels
	 * \param[in] bool exchange_chan change channel or not
	 * \param[in] int filling_method the filling method, see marcos like BORDER_CONSTANT in OpenCV
	 * \param[in] int cval pixel Scalar for constatn padding
	 * \param[in] bool v_flip use vertical flip or not
	 * \param[in] bool h_flip use horizontal flip or not
	 * \param[in] vector<shared_ptr<Noise>> noises noises to apply
	 * \param[in] bool resize use resize or not
	 * \param[in] double rsz_min minimun proportion
	 * \param[in] double rsz_max maximun proportion
	*/
	ImageGenerator(bool hist_equlize, bool mean_normalization,
				   bool rotate, double rotate_min, double rotate_max,
				   bool v_move, int v_move_min, int v_move_max, 
				   bool h_move, int h_move_min, int h_move_max,
				   bool exchange_chan, int filling_method, int cval,
				   bool v_flip, bool h_flip,
				   std::vector<std::shared_ptr<Noise>> noises,
				   bool resize, double rsz_min, double rsz_max);

	// Destructor
	~ImageGenerator();

	// member function
	/* \brief
	    This member function will generate images
		.. and store the result in the res vector.
		If you want more different images, try using this function
		.. one more time.(Though, some effect is fixed and stay unchange).
	 * \param[in] Mat& src src image
	 * \param[in] std::vector<cv::Mat> &res to store the generated images, 
		.. and the first image is the original one.
	*/
	void gen(cv::Mat &src, std::vector<cv::Mat> &res);

	/* \brief
		A debug function, which just tests the tool function
	*/
	void debug_gen(cv::Mat &src, std::vector<cv::Mat> &res);

// private:
private:
	// data members
	bool hist_eqlize_;		// use histogram equalization or not
	// bool ZCA_whiten_;		// use ZCA whiten or not, TODO
	bool mean_normalization_; //use mean normalization or not
	// bool variance_normalization_; use variance normalization or not, TODO
	bool rotate_;
	double rtt_min_;		// min rotate rate, must be bigger than 0
	double rtt_max_;		// max rotate rate, must be smaller than 360.0
	bool v_move_;
	int v_min_;		// min vertical move steps, suggest to be larger than -(width / 2)
	int v_max_;		// max vertical move steps, suggest to be smaller than (width / 2)
	bool h_move_;
	int h_min_;		// min horizontal move steps, suggest to be larger than -(width / 2)
	int h_max_;		// max horizontal move steps, suggest to be smaller than (width / 2)
	bool exchange_chan_;	// use channel exchange or not
	int filling_method_;	// filling method for move, rotate operation, use cv::BORDER_xxxx
	int cval_;				// constant for constfilling method
	bool v_flip_;			// use vertical flip or not
	bool h_flip_;			// use horizontal flip or not
	std::vector<std::shared_ptr<Noise>> noises_;	// user-defined noises
	bool resize_;
	double rsz_min_;			// min resize proportion, must be larger then 0
	double rsz_max_;			// max resize proportion, must be larger then 0

	// private tools
	/* \brief 
		HistEqualize will apply histogram equalize to the src image
		..and then push the result to the res vector.
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
	*/
	void HistEqualize(cv::Mat &src, std::vector<cv::Mat> &res);

	/* \brief
		ZCAWhiten will apply ZCA whiten operation to the src image
		..and then push the result to the res vector.
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
	*/
	// void ZCAWhiten(cv::Mat &src, std::vector<cv::Mat> &res);

	/* \brief
		MeanNormalize will apply mean normalization operation to the src image
		..and then push the result to the res vector.
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
	*/
	void MeanNormalize(cv::Mat &src, std::vector<cv::Mat> &res);

	/* \brief
		Rotate will apply rotate operation with random angle
		..between r_min & r_max to the src image
		..and then push the result to the res vector.
		If you dont want to rotate the image, set r_min > r_max
		ATTENTION: r_min == r_max has other usage.
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
		\param[in] double r_min minimun of the rotate angle
		\param[in] double r_max maximun of the rotate angle
	*/
	void Rotate(cv::Mat &src, std::vector<cv::Mat> &res, double r_min, double r_max);

	/* \brief
		Move will apply move operation with random steps
		..vertically between v_min & v_max  to the src image
		..and then push the result to the res vector.
		If you dont want to move the image vertically, set v_min > v_max.
		ATTENTION: v_min == v_max have other usage
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
		\param[in] int v_min minimun of the vertical step
		\param[in] int v_max maximun of the vertical step
	*/
	void VMove(cv::Mat &src, std::vector<cv::Mat> &res, int v_min, int v_max);

	/* \brief
		Move will apply move operation with random steps
		..horizontally between h_min & h_max  to the src image
		..and then push the result to the res vector.
		If you dont want to move the image horizontally, set h_min > h_max.
		ATTENTION: h_min == h_max have other usage
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
		\param[in] int h_min minimun of the horizontal step
		\param[in] int h_max maximun of the horizontal step
	*/
	void HMove(cv::Mat &src, std::vector<cv::Mat> &res, int h_min, int h_max);

	/* \brief
		ChangeChannel will exchange the RGB channel of
		..the source image to create new image.
		It will convert RGB to RBG, GBR, GRB, BRG, BGR,
		..which mean it will create 5 new images and push them into vector.
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
	*/
	void ChangeChannel(cv::Mat &src, std::vector<cv::Mat> &res);
	/* \brief
		ShiftChannel is a helper function of ChangeChannel
		It shift all the channel to its right. (Well, in RGB form, not BGR)
		\param[in] cv::Mat &src source Mat
		\param[in] cv::Mat &dst destination Mat
	*/
	void ShiftChannel(cv::Mat &src, cv::Mat &dst);

	/* \brief
		VFlip will vertically flip the source image to create a new one,
		..then push it to the res vector
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
	*/
	void VFlip(cv::Mat &src, std::vector<cv::Mat> &res);

	/* \brief
		HFlip will horizontally flip the source image to create a new one,
		..then push it to the res vector
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
	*/
	void HFlip(cv::Mat &src, std::vector<cv::Mat> &res);

	/* \brief
		ApplyNosie will apply the noise in vector noises to the image
		..sequtially, the amount of new images depend on the noise 
		..functions when you init the generator.
		If the length of noises is 0, it wont do anything.
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
		\param[in] std::vector<std::shared_ptr<Noise>>& noises vector to store noises
	*/
	void ApplyNoise(cv::Mat &src, std::vector<cv::Mat> &res, std::vector<std::shared_ptr<Noise>>& noises);

	/* \brief
		Resize randomly resize to the scale between rsz_min & rsz_max.
		If you dont want to use he resize, set rsz_min > rsz_max
		ATTENTION: rsz_min == rsz_max has other usage
		\param[in] cv::Mat &src source Mat
		\param[in] std::vector<cv::Mat> &res vector to store result
		\param[in] std::vector<std::function<void(cv::Mat&)>>& noises vector to store noises
	*/
	void Resize(cv::Mat &src, std::vector<cv::Mat> &res, double rsz_min, double rsz_max);

	// TODO: shear operation
};

