#include <iostream>
#include <algorithm>

#include "optimal.h"

using namespace std;

void optimal(opt &ret, vector<vector<float>>& mydata, vector<float>& weights) {

	vector<float> data = mydata[0];
	vector<int> index(data.size());
	_sort(data, index);

	vector<int> y;
	vector<float> weights1;
	for(int i=0; i<index.size(); i++) {
		y.push_back(mydata[2][index[i]]);
		weights1.push_back(weights[index[i]]);
	}

	vector<int> total_pos_idx;
	vector<int> total_neg_idx;
	_find(y, total_pos_idx, total_neg_idx);

	float T_plus = 0.0;
	float T_minus = 0.0;
	_sum(T_plus, weights1, total_pos_idx);
	_sum(T_minus, weights1, total_neg_idx);


	vector<float> error_list;
	vector<int> polarity;
	for(int i=0; i<mydata[0].size(); i++) {

		float S_plus = 0;
		float S_minus = 0;
		_sum_x_gt_idx(S_plus, weights1, total_pos_idx, i);
		_sum_x_gt_idx(S_minus, weights1, total_neg_idx, i);

		float a = S_plus + (T_minus - S_minus);
		float b = S_minus + (T_plus - S_plus);
		if(a <= b) {
			error_list.push_back(a);
			polarity.push_back(1);
		}
		else {
			error_list.push_back(b);
			polarity.push_back(2);
		}
	}

	float min_error_idx = 0;
	for(int i=1; i<error_list.size(); i++)
		if(error_list[i] < error_list[min_error_idx])
			min_error_idx = i;

	ret.bestx = data[min_error_idx];
	ret.min_error = error_list[min_error_idx];
	ret.polarity = polarity[min_error_idx];
}

void _sort(vector<float> &data, vector<int> &index) {

	size_t n(0);
	generate(index.begin(), index.end(), [&]{return n++;});
	sort(index.begin(), index.end(), [&](int a, int b){return data[a] < data[b];});
	sort(data.begin(), data.end());
}

void _find(const vector<int> &y, vector<int> &total_pos_idx, vector<int> &total_neg_idx) {

	for(int i=0; i<y.size(); i++) {
		if(y[i] == 1)
			total_pos_idx.push_back(i);
		else if(y[i] == -1)
			total_neg_idx.push_back(i);
		else
			cout << "ERROR(_find): y[i] value not equal to +-1 when i = " << i << "";
	}
}

void _sum(float &T, const vector<float> &weights1, const vector<int> &total_idx) {

	for(int i=0; i<total_idx.size(); i++)
		T += weights1[total_idx[i]];
}

void _sum_x_gt_idx(float &S, const vector<float> &weights1, const vector<int> &total_idx, int x) {

	for(int i=0; i<total_idx.size(); i++)
		if(x > total_idx[i])
			S += weights1[total_idx[i]];
}
