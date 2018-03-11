#ifndef __OPTIMAL_H__
#define __OPTIMAL_H__

#include "common.h"
#include <vector>

using namespace std;

void optimal(opt &ret, vector<vector<float>> mydata, vector<float> weights);

void _sort(vector<float> &data, vector<int> &index);

void _find(const vector<int> &y, vector<int> &total_pos_idx, vector<int> &total_neg_idx);

void _sum(float &T, const vector<float> &weights1, const vector<int> &total_idx);

void _sum_x_gt_idx(float &S, const vector<float> &weights1, const vector<int> &total_idx, int x);

#endif /* __OPTIMAL_H__ */
