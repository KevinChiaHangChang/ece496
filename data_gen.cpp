#include <common/xf_common.h>
// #include <common/xf_sw_utils.h>
//#include <imgproc/xf_resize.hpp>
//#include <imgproc/xf_hist_equalize.hpp>
//#include <imgproc/xf_integral_image.hpp>
//#include <core/xf_mean_stddev.hpp>
#ifdef __SDSCC__
#undef __ARM_NEON__
#undef __ARM_NEON
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define __ARM_NEON__
#define __ARM_NEON
#else
#endif

#include <vector>

#include "data_gen.h"

using namespace cv;

//#ifdef __cplusplus
//extern "C" {
//#endif

// std::vector of first 10 face images + non face images
const char* face_filenames[] = {"1.pgm","2.pgm","3.pgm","4.pgm","5.pgm","6.pgm","7.pgm","8.pgm","9.pgm","10.pgm"};
const char* non_face_filenames[] = {"1.png","2.png","3.png","4.png","5.png","6.png","7.png","8.png","9.png","10.png"};

#define NUM_ROWS	24
#define NUM_COLS	24

void data_gen(std::vector<std::vector<float>>& face_data, std::vector<std::vector<float>>& non_face_data, const int rows, const int cols, const std::vector<int>& A) {

	// convert input Haar feature vector into cv::Mat
	cv::Mat haar = cv::Mat(A).reshape(rows,cols);

	// iterate over face images
	for (int i = 0; i < 10; i++) {

		// load image by filename
		char filename[] = "face/";
		strcat(filename,face_filenames[i]);
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> img = xf::imread(filename,XF_NPPC1);
		cv::Mat myImg(NUM_ROWS, NUM_COLS, CV_8SC1);
		myImg = cv::imread(filename);

		// resize image
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> resize = xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1>::zeros(24,24);
		// xf::resize(img,resize);
		cv::Mat resized;
		cv::resize(myImg,resized,cv::Size(NUM_ROWS,NUM_COLS));

		// histogram equalization
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> histeq = xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1>::zeros(24,24);
		// xf::equalizeHist(resize,histeq);
		cv::Mat histeq;
		cv::equalizeHist(resized,histeq);

		// compute integral image
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> integral = xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1>::zeros(24,24);
		// xf::integral(histeq,integral);
		cv::Mat integral;
		cv::integral(histeq,integral);

		// apply Haar feature filter
		cv::Mat filter;
		cv::filter2D(integral,filter,-1,haar);

		// compute mean
		// unsigned short mean;
		// unsigned short stddev;
		// xf::meanStdDev(integral,&mean,&stddev);
		double mean = cv::mean(filter)[0];

		// update face_data
		face_data[0][i] = (float) mean;
		face_data[1][i] = 1;
		face_data[2][i] = 1;

	}

	// iterate over non face images
	for (int i = 0; i < 10; i++){

		// load image by filename
		char filename[] = "nonface/";
		strcat(filename,non_face_filenames[i]);
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> img = xf::imread(filename,XF_NPPC1);
		cv::Mat myImg(NUM_ROWS, NUM_COLS, CV_8SC1);
		myImg = cv::imread(filename);

		// resize image
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> resize = xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1>::zeros(24,24);
		// xf::resize(img,resize);
		cv::Mat resized;
		cv::resize(myImg,resized,Size(NUM_ROWS,NUM_COLS));

		// histogram equalization
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> histeq = xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1>::zeros(24,24);
		// xf::equalizeHist(resize,histeq);
		cv::Mat histeq;
		cv::equalizeHist(resized,histeq);

		// compute integral image
		// xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1> integral = xf::Mat<XF_8UC1,NUM_ROWS,NUM_COLS,XF_NPPC1>::zeros(24,24);
		// xf::integral(histeq,integral);
		cv::Mat integral;
		cv::integral(histeq,integral);

		// apply Haar feature filter
		cv::Mat filter;
		cv::filter2D(integral,filter,-1,haar);

		// compute mean
		// unsigned short mean;
		// unsigned short stddev;
		// xf::meanStdDev(integral,&mean,&stddev);
		double mean = cv::mean(filter)[0];

		// update face_data
		non_face_data[0][i] = (float) mean;
		non_face_data[1][i] = 1;
		non_face_data[2][i] = 1;
	}
}

//#ifdef __cplusplus
//}
//#endif
