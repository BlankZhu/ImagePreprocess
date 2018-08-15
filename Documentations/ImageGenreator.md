DESCRIPTION:
ImageGenreator is a class that help you preprocess the images in order to get
.. more training data for your neural networks.

OPERATIONS:
  1. Histogram Equalization.
    (This may help you get a better perform image)
  2. ZCA Whitening* (TODO)
    (Helps to get a ZCA Whitened image)
  3. Mean normalization
    (Helps to get a mean normalized image)
  4. Variance Normalization* (TODO)
    (Helps to get a variance normalized image)
  5. Random Rotation
    (Randomly rotate the image by given range)
  6. Random Vertical Moving
    (Randomly move the image vertically)
  7. Random Horizontally Moving
    (Randomly move the image horizaontally by given range)
  8. Channel Exchange
    (Exchange the channels of a image randomly)
  9. Vertical Flip
    (Randomly flip the image vertically)
  10. Horizontal Flip
    (Randomly flip the image horizontally)
  11. Noise
    (Add random noise to the image)
  12. Resize
    (Randomly resize the image by given range)
  13. *(More TODO, for example, slide window)
    (Placeholder...)

APIs:
FUNTIONS:
  1. Default Constructor
    ImageGenerator::ImageGenerator()
    This constructor initialize a default ImageGenerator.
		It WILL apply Histogram Equalization.
		It WILL apply Mean Normalization.
		It WILL randomly rotate the image in [0, 360.0). (degree)
		It WILL vertically move the image in [-50, 50). (pixels)
		It WILL horizontally move the image in [-50, 50). (pixels)
		It WILL exchange the RGB channels.
		It WILL use zero padding as default filling method.
		It WILL NOT add any noise on the image.
		It WILL resize the image in [0.5, 1.5). (proportion)
  2. Other Constructor
    ImageGenerator(bool hist_equlize, bool mean_normalization,
				   bool rotate, double rotate_min, double rotate_max,
				   bool v_move, int v_move_min, int v_move_max, 
				   bool h_move, int h_move_min, int h_move_max,
				   bool exchange_chan, int filling_method, int cval,
				   bool v_flip, bool h_flip,
				   std::vector<std::function<void(cv::Mat&)>> noises,
				   bool resize, double rsz_min, double rsz_max);
    This constructor will initialize a fixed ImageGenerator.
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
      ..For the filling method, see marcos like BORDER_CONSTANT in OpenCV.
		It WILL add noises to image.
			..if you dont want to add noise, use an empty vector
		It WILL resize the image in [rsz_min, rsz_max).
			..if you dont want to resize, set resize == false
  3. void ImageGenerator::gen(Mat& input_mat, int gen_amount, vecter<Mat>& res)
    This member function will generate given amount of mat from input_mat,
    .. and store the result in the res vector.
