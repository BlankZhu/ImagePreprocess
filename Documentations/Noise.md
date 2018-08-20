DESCRIPTION:
Noise is a base class for a series of classes,
like gaussian noise, salt noise, pepper noise and periodic noise.
Currently there are these 4 noises, but you can add your own noise
..base on the class Noise.

OPERATIONS:
  1. Gaussian Noise
    It will apply gaussian noise to your Mat,
    ..you can specify the mean and stddev.
  2. Salt Noise
    It will apply salt noise to your Mat,
    ..you can specify the probability.
  3. Pepper Noise
    It will apply pepper noise to your Mat,
    ..you can specify the probability.
  4. Periodic Noise
    It will apply periodic noise, which is base on sin, to your Mat,
    ..you can specify the period, intensity, and direction(v or h).

APIs:
FUNTIONS:
  1. Noise()
    Default constructor.
    However, Noise itself is an abstract class.
    Check the comments in the source code to see 
    ..the constructor of other class derived from Noise
  2. ApplyNoise(cv::Mat &src, cv::Mat &dst)
    This function will apply the noise to the source mat.
    However, this function is pure virtual in Noise,
    ..so you need to override it if you want to use your
    ..own class.