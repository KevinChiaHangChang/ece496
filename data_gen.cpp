//#include <common/xf_common.h>
//#include <imgproc/xf_resize.hpp>
//#include <imgproc/xf_hist_equalize.hpp>
//#include <imgproc/xf_integral_image.hpp>
//#include <core/xf_mean_stddev.hpp>
//
//#include "data_gen.h"
//
//using namespace std;
//
//// vector of first 10 face images + non face images
//const char* face_filenames[] = {"1.pgm","2.pgm","3.pgm","4.pgm","5.pgm","6.pgm","7.pgm","8.pgm","9.pgm","10.pgm"};
//const char* non_face_filenames[] = {"1.png","2.png","3.png","4.png","5.png","6.png","7.png","8.png","9.png","10.png"};
//
//void data_gen(vector<data>& face_data, vector<data>& non_face_data, const xf::Mat<int16_t> A) {
//
//	// iterate over face images
//	for (int i = 0; i < 10; i++) {
//
//		// load image by filename
//		char filename[] = "face/";
//		strcat(filename,face_filenames[i]);
//		xf::Mat<XF_8UC1> img = xf::imread(filename);
//
//		// resize image
//		xf::Mat<XF_8UC1> resize = xf::Mat<XF_8UC1>::zeros(24,24);
//		xf::resize(img,resize);
//
//		// histogram equalization
//		xf::Mat<XF_8UC1> histeq = xf::Mat<XF_8UC1>::zeros(24,24);
//		xf::equalizeHist(resize,histeq);
//
//		// compute integral image
//		xf::Mat<XF_8UC1> integral = xf::Mat<XF_8UC1>::zeros(24,24);
//		xf::integral(histeq,integral);
//
//		// apply Haar feature filter
//		// TODO
//
//		// compute mean
//		unsigned short mean;
//		unsigned short stddev;
//		xf::meanStdDev(integral,&mean,&stddev);
//
//		// update face_data
//		face_data[i].avg = mean;
//		face_data[i].target = 1;
//		face_data[i].polarity = 1;
//
//	}
//
//	// iterate over non face images
//	for (int i = 0; i < 10; i++){
//
//		// load image by filename
//		char filename[] = "nonface/";
//		strcat(filename,non_face_filenames[i]);
//		xf::Mat<XF_8UC1> img = xf::imread(filename);
//
//		// resize image
//		xf::Mat<XF_8UC1> resize = xf::Mat<XF_8UC1>::zeros(24,24);
//		xf::resize(img,resize);
//
//		// histogram equalization
//		xf::Mat<XF_8UC1> histeq = xf::Mat<XF_8UC1>::zeros(24,24);
//		xf::equalizeHist(resize,histeq);
//
//		// compute integral image
//		xf::Mat<XF_8UC1> integral = xf::Mat<XF_8UC1>::zeros(24,24);
//		xf::integral(histeq,integral);
//
//		// apply Haar feature filter
//		// TODO
//
//		// compute mean
//		unsigned short mean;
//		unsigned short stddev;
//		xf::meanStdDev(integral,&mean,&stddev);
//
//		// update face_data
//		non_face_data[i].avg = mean;
//		non_face_data[i].target = 1;
//		non_face_data[i].polarity = 1;
//	}
//}
