// Preprocess.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

int main()
{
	ImageGenerator img_gen;
	cv::Mat src;
	std::vector<cv::Mat> res;

	// load source image
	src = cv::imread("./imgs/lena.jpg");
	if (src.empty())
	{
		std::cout << "Not loaded!";
		return -1;
	}

	// generate image, DEBUGGING
	img_gen.debug_gen(src, res);

	// see the amount of res
	std::cout << "Gen: " << res.size() << std::endl;

	// show
	cv::imshow("source", src);
	for (size_t i = 0; i < res.size(); ++i)
	{
		cv::imshow(std::to_string(i) + "transed", res[i]);
		if (i >= 30)
		{
			std::cout << "first 30 images shown" << std::endl;
			break;
		}
	}

	cv::waitKey();
}
