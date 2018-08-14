DESCRIPTION:
ImageGenreator is a class that help you preprocess the images in order to get
.. more training data for your neural networks.

OPERATIONS:
  1. Histogram Equalization.
    (This may help you get a better perform image)
  2. ZAC Whitening
    (Helps to get a ZAC Whitened image)
  3. Random Rotation
    (Randomly rotate the image by given range)
  4. Random Vertical Moving
    (Randomly move the image vertically)
  5. Random Horizontally Moving
    (Randomly move the image horizaontally by given range)
  6. Random Channel Exchange
    (Exchange the channels of a image randomly)
  7. Border filling
    (Fill the pixels outside the border by specifed method)
  8. Random Vertical Flip
    (Randomly flip the image vertically)
  9. Random Horizontal Flip
    (Randomly flip the image horizontally)
  10. Random Noise
    (Add random noise to the image)
  11. Resize
    (Randomly resize the image by given range)
  12. (More TODO, for example, slide window)
    (Placeholder...)

APIs:
FUNTIONS:
  1. Default Constructors
    ImageGenerator::ImageGenerator()
    This constructor initialize a default ImageGenerator.
    It WILL apply Histogram Equalization.
    It WILL apply ZAC Whitening.
    It WILL randomly rotate the image in [0, 2pi).
    It WILL vertically move the image in [- (max_height) / 2, max_height / 2).
    It WILL horizontally move the image in [- (max_width) / 2, max_width / 2).
    It WILL exchange the channels randomly.
    It WILL use zero padding as default filling method.
    It WILL NOT add any noise on the image
    It WILL resize the image in [0.5, 1)
  2. ImageGenerator::ImageGenerator(bool hist_eqlize, bool ZAC_whiten, 
                                    double rotate_min, double rotate_max,
                                    int v_move_min, int v_move_max,
                                    int h_move_min, int h_move_max,
                                    bool exchange_chan,
                                    int filling_method, int cval,
                                    vector<function<(void(Mat&, Mat&))>> noises,
                                    double rsz_min, double rsz_max)
    This constructor initialize a fixed ImageGenerator.
    It may apply Histogram Equalization according to hist_eqlize.
    It may apply ZAC Whitening according to ZAC_whiten.
    It WILL randomly rotate the image in [rotate_min, rotate_max).
        ..if you dont want to rotate, set rotate_min == rotate_max
    It WILL vertically move the image in [v_move_min, v_move_max).
        ..if you dont want to move, set v_move_min == v_move_max
    It WILL horizontally move the image in [h_move_min, h_move_max).
        ..if you dont want to move, set h_move_min == h_move_max
    It may exchange the channels randomly according to exchange_chan.
    It WILL use filling method specifed by filling_method.
        ..if you use constant filling, you should specify cval,
        ..otherwise the cval is useless, you can set a lucky number you like.
    It WILL add noises to image.
        ..if you dont want to add noise, use an empty vector
    It WILL resize the image in [rsz_min, rsz_max).
        ..if you dont want to resize, set rsz_min == rsz_max
  3. void ImageGenerator::gen(Mat& input_mat, int gen_amount, vecter<Mat>& res)
    This member function will generate given amount of mat from input_mat,
    .. and store the result in the res vector.

CONSTANTS:
    kPI = CV::pi (3.1415926535897)
    kZERO_PADDING = 0
    kCONSTANT_PADDING = 1
    kNEAREST_PADDING = 2