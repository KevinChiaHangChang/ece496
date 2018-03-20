#ifndef __ADABOOST_H__
#define __ADABOOST_H__

#include "common_data.h"
#include <vector>

using namespace std;

void adaboost();

void _update_weights(	const vector<vector<float>>& face_data,
						const vector<vector<float>>& non_face_data,
						vector<float>& face_weights,
						vector<float>& non_face_weights,
						const float& theta,
						const float& classifier_beta,
						const int& classifier_polarity	);

void _normalize_weights(vector<float>& weights);

int _find_min_idx(const vector<float>& error);

#endif /* __ADABOOST_H__ */
