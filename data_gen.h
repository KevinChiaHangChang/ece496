#ifndef __DATA_GEN_H__
#define __DATA_GEN_H__

#ifdef __SDSCC__
#undef __ARM_NEON__
#undef __ARM_NEON
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#define __ARM_NEON__
#define __ARM_NEON
#else
#endif
#include <vector>

using namespace std;

//#ifdef __cplusplus
//extern "C" {
//#endif

void data_gen(std::vector<std::vector<float>>& face_data, std::vector<std::vector<float>>& non_face_data, const int rows, const int cols, const std::vector<int>& A);

//#ifdef __cplusplus
//}
//#endif

#endif /* __DATA_GEN_H__ */
