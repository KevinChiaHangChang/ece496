#ifndef __DATA_GEN_H__
#define __DATA_GEN_H__

using namespace std;

class data {
	unsigned short avg;
	signed short target;
	signed short polarity;
};

void data_gen(vector<opt>& face_data, vector<opt>& non_face_data, const xf::Mat<float> A);

#endif /* __DATA_GEN_H__ */
