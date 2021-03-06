// Preprocess.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"


std::string
GenerateNewName(std::string filename_without_ext, size_t i, std::string ext)
{
	return filename_without_ext + std::to_string(i) + "." + ext;
}


void listFiles(const char * dir)
{
	using namespace std;

	char dirNew[200];
	strcpy_s(dirNew, dir);
	strcat_s(dirNew, "\\*.*");    // 在目录后面加上"\\*.*"进行第一次搜索

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dirNew, &findData);
	if (handle == -1)        // 检查是否成功
		return;

	do
	{
		if (findData.attrib & _A_SUBDIR)	// get dir
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;

			cout << findData.name << "\t<dir>\n";

			// 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
			strcpy_s(dirNew, dir);
			strcat_s(dirNew, "\\");
			strcat_s(dirNew, findData.name);

			listFiles(dirNew);
		}
		else    // get files
		{
			std::string path(dir);

			if (std::regex_match(findData.name, std::regex(".*\.jpg")))
			{
				cout << findData.name << "\t";
				string tmp_s(findData.name);
				string name_without_ext = string(tmp_s.begin(), tmp_s.end() - 4);

				// check if xml exist too
				tmp_s = name_without_ext + ".xml";
				tmp_s = path + "/" + tmp_s;
				if ((_access(tmp_s.c_str(), 0) != -1))
				{
					std::shared_ptr<Noise> n1(new GaussianNoise());
					std::shared_ptr<Noise> n2(new PeriodicNoise(false, 128 * CV_PI, 16));
					std::vector<std::shared_ptr<Noise>> n({ n1,n2});

					// both jpg and xml exist, start operation
					ImageGenerator generator(true, true, false, 0, 0, false, 0, 0, false, 0, 0, true, 0, 0, false, false, n, false, 0, 0);
					std::vector<cv::Mat> res;
					cv::Mat src = cv::imread(path + "/" + findData.name);
					generator.gen(src, res);

					// copy, rename xml file,
					//.. and create new jpg file
					for (size_t i = 0; i < res.size(); ++i)
					{
						// copy xml
						ifstream inFile(tmp_s, ios::binary);
						ofstream outFile(path + "/" + GenerateNewName(name_without_ext + '-', i, "xml"), ios::binary);

						while (true)  // Though perhaps this condition is wrong
						{
							auto c = inFile.get();
							if (c != EOF)
								outFile.put(c);
							else
								break;
						}

						inFile.close();
						outFile.close();

						// write jpg
						cv::imwrite(path + "/" + GenerateNewName(name_without_ext + '-', i, "jpg"), res[i]);
					}
				}
				cout << "Convertion finished." << std::endl;;
			}
		}
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);    // 关闭搜索句柄
}

int main()
{
	ImageGenerator img_gen;
	cv::Mat src;
	std::vector<cv::Mat> res;

	listFiles("./Convertion");

	cv::waitKey();
}
