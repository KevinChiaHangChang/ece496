#include <common/xf_common.h>
#include <algorithm>
#include <iterator>

#include "adaboost.h"
#include "data_gen.h"
#include "optimal.h"
// #include "common.h"

using namespace std;

// for now, use only 10 faces + non-faces
// #define NUM_FACES			200
// #define NUM_NON_FACES		400
#define NUM_FACES			10
#define NUM_NON_FACES		10
#define NUM_HAAR_FEATURES	10
#define NUM_CASCADES		3

// define Haar feature matrices
//xf::Mat<int> A0 = (xf::Mat<int,6,3> << -1,-1,-1,/**/-1,-1,-1,/**/-1,-1,-1,/**/1,1,1,/**/1,1,1,/**/1,1,1);
//xf::Mat<int> A1 = (xf::Mat<int,6,6> << -1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,/**/1,1,1,1,1,1,/**/1,1,1,1,1,1,/**/1,1,1,1,1,1);
//xf::Mat<int> A2 = (xf::Mat<int,6,9> << -1,-1,-1,-1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,-1,-1,-1,/**/1,1,1,1,1,1,1,1,1,/**/1,1,1,1,1,1,1,1,1,/**/1,1,1,1,1,1,1,1,1);
//xf::Mat<int> A3 = (xf::Mat<int,8,4> << -1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1);
//xf::Mat<int> A4 = (xf::Mat<int,8,6> << -1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1);
//xf::Mat<int> A5 = (xf::Mat<int,8,4> << -1,-1,-1,-1,/**/-1,-1,-1,-1,/**/-1,-1,-1,-1,/**/-1,-1,-1,-1,/**/-1,-1,-1,-1,/**/1,1,1,1,/**/1,1,1,1,/**/1,1,1,1,/**/1,1,1,1);
//xf::Mat<int> A6 = (xf::Mat<int,8,4> << 1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1);
//xf::Mat<int> A7 = (xf::Mat<int,8,6> << 1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1);
//xf::Mat<int> A8 = (xf::Mat<int,8,7> << 1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1);
//xf::Mat<int> A9 = (xf::Mat<int,8,9> << 1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1);
//xf::Mat<int> A[NUM_HAAR_FEATURES] = {A0,A1,A2,A3,A4,A5,A6,A7,A8,A9};
//vector<vector<int>> A0 = {	{-1,-1,-1},
//							{-1,-1,-1},
//							{-1,-1,-1},
//							{1,1,1},
//							{1,1,1},
//							{1,1,1}	};
//vector<vector<int>> A1 = {	{-1,-1,-1,-1,-1,-1},
//							{-1,-1,-1,-1,-1,-1},
//							{-1,-1,-1,-1,-1,-1},
//							{1,1,1,1,1,1},
//							{1,1,1,1,1,1},
//							{1,1,1,1,1,1}	};
//vector<vector<int>> A2 = {	{-1,-1,-1,-1,-1,-1,-1,-1,-1},
//							{-1,-1,-1,-1,-1,-1,-1,-1,-1},
//							{-1,-1,-1,-1,-1,-1,-1,-1,-1},
//							{1,1,1,1,1,1,1,1,1},
//							{1,1,1,1,1,1,1,1,1},
//							{1,1,1,1,1,1,1,1,1}	};
//vector<vector<int>> A3 = {	{-1,-1,1,1},
//							{-1,-1,1,1},
//							{-1,-1,1,1},
//							{-1,-1,1,1},
//							{-1,-1,1,1},
//							{-1,-1,1,1},
//							{-1,-1,1,1},
//							{-1,-1,1,1}	};
//vector<vector<int>> A4 = {	{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1},
//							{-1,-1,-1,1,1,1}	};
//vector<vector<int>> A5 = {	{-1,-1,-1,-1},
//							{-1,-1,-1,-1},
//							{-1,-1,-1,-1},
//							{-1,-1,-1,-1},
//							{1,1,1,1},
//							{1,1,1,1},
//							{1,1,1,1},
//							{1,1,1,1}	};
//vector<vector<int>> A6 = {	{1,-1,-1,1},
//							{1,-1,-1,1},
//							{1,-1,-1,1},
//							{1,-1,-1,1},
//							{1,-1,-1,1},
//							{1,-1,-1,1},
//							{1,-1,-1,1},
//							{1,-1,-1,1}	};
//vector<vector<int>> A7 = {	{1,1,-1,-1,1,1},
//							{1,1,-1,-1,1,1},
//							{1,1,-1,-1,1,1},
//							{1,1,-1,-1,1,1},
//							{1,1,-1,-1,1,1},
//							{1,1,-1,-1,1,1}	};
//vector<vector<int>> A8 = {	{1,1,-1,-1,-1,1,1},
//							{1,1,-1,-1,-1,1,1},
//							{1,1,-1,-1,-1,1,1},
//							{1,1,-1,-1,-1,1,1},
//							{1,1,-1,-1,-1,1,1},
//							{1,1,-1,-1,-1,1,1},
//							{1,1,-1,-1,-1,1,1}	};
//vector<vector<int>> A9 = {	{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1},
//							{1,1,1,-1,-1,-1,1,1,1}	};
//vector<vector<vector<int>>> A = {A0,A1,A2,A3,A4,A5,A6,A7,A8,A9};

vector<int> A0 = {	-1,-1,-1,
					-1,-1,-1,
					-1,-1,-1,
					1,1,1,
					1,1,1,
					1,1,1	};
vector<int> A1 = {	-1,-1,-1,-1,-1,-1,
					-1,-1,-1,-1,-1,-1,
					-1,-1,-1,-1,-1,-1,
					1,1,1,1,1,1,
					1,1,1,1,1,1,
					1,1,1,1,1,1	};
vector<int> A2 = {	-1,-1,-1,-1,-1,-1,-1,-1,-1,
					-1,-1,-1,-1,-1,-1,-1,-1,-1,
					-1,-1,-1,-1,-1,-1,-1,-1,-1,
					1,1,1,1,1,1,1,1,1,
					1,1,1,1,1,1,1,1,1,
					1,1,1,1,1,1,1,1,1	};
vector<int> A3 = {	-1,-1,1,1,
					-1,-1,1,1,
					-1,-1,1,1,
					-1,-1,1,1,
					-1,-1,1,1,
					-1,-1,1,1,
					-1,-1,1,1,
					-1,-1,1,1	};
vector<int> A4 = {	-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1,
					-1,-1,-1,1,1,1	};
vector<int> A5 = {	-1,-1,-1,-1,
					-1,-1,-1,-1,
					-1,-1,-1,-1,
					-1,-1,-1,-1,
					1,1,1,1,
					1,1,1,1,
					1,1,1,1,
					1,1,1,1	};
vector<int> A6 = {	1,-1,-1,1,
					1,-1,-1,1,
					1,-1,-1,1,
					1,-1,-1,1,
					1,-1,-1,1,
					1,-1,-1,1,
					1,-1,-1,1,
					1,-1,-1,1	};
vector<int> A7 = {	1,1,-1,-1,1,1,
					1,1,-1,-1,1,1,
					1,1,-1,-1,1,1,
					1,1,-1,-1,1,1,
					1,1,-1,-1,1,1,
					1,1,-1,-1,1,1	};
vector<int> A8 = {	1,1,-1,-1,-1,1,1,
					1,1,-1,-1,-1,1,1,
					1,1,-1,-1,-1,1,1,
					1,1,-1,-1,-1,1,1,
					1,1,-1,-1,-1,1,1,
					1,1,-1,-1,-1,1,1,
					1,1,-1,-1,-1,1,1	};
vector<int> A9 = {	1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1,
					1,1,1,-1,-1,-1,1,1,1	};
vector<vector<int>> A = {A0,A1,A2,A3,A4,A5,A6,A7,A8,A9};
vector<int> rows = {6,6,6,8,8,8,8,8,8,8};
vector<int> cols = {3,6,9,4,6,4,4,6,7,9};

void adaboost() {

	// initialize weights
	std::vector<float> face_weights(NUM_FACES,1/(2*NUM_FACES));
	std::vector<float> non_face_weights(NUM_NON_FACES,1/(2*NUM_NON_FACES));

	// main loop
	for (int i = 0; i < NUM_CASCADES; i++){

		// normalize weights
		_normalize_weights(face_weights);
		_normalize_weights(non_face_weights);

		// initialize error, beta, threshold, polarity vectors
		vector<float> error(NUM_HAAR_FEATURES,0);
		vector<float> beta(NUM_HAAR_FEATURES,0);
		vector<float> threshold(NUM_HAAR_FEATURES,0);
		vector<float> polarity(NUM_HAAR_FEATURES,0);

		// loop over Haar features
		for (int j = 0; j < NUM_HAAR_FEATURES; j++) {

			// generate data
			vector<vector<float>> face_data(3,vector<float>(NUM_FACES));
			vector<vector<float>> non_face_data(3,vector<float>(NUM_NON_FACES));
			// data_gen(face_data,non_face_data,rows[j],cols[j],A[j]);

			// find minimum error, threshold, polarity
			vector<vector<float>> my_data = face_data;
			my_data[0].insert(face_data[0].end(),non_face_data[0].begin(),non_face_data[0].end());
			my_data[1].insert(face_data[1].end(),non_face_data[1].begin(),non_face_data[1].end());
			my_data[2].insert(face_data[2].end(),non_face_data[2].begin(),non_face_data[2].end());
			vector<float> weights = face_weights;
			weights.insert(weights.end(),non_face_weights.begin(),non_face_weights.end());
			opt extract;
			optimal(extract,my_data,weights);

			// update error
			error[j] = extract.min_error;
			// update beta
			beta[j] = error[j]/(1-error[j]);
			// update threshold
			threshold[j] = extract.bestx;
			// update polarity
			polarity[j] = extract.polarity;

		}

		// form + define classifier
		int min_idx = _find_min_idx(error);
		float theta = threshold[min_idx];
		vector<int> feature = A[min_idx];
		int classifier_polarity = polarity[min_idx];
		float classifier_beta = beta[min_idx];
		float alpha = log(1/classifier_beta);

		// update weights of misclassified points
		vector<vector<float>> best_face_data(3,vector<float>(NUM_FACES));
		vector<vector<float>> best_non_face_data(3,vector<float>(NUM_NON_FACES));
		// data_gen(best_face_data,best_non_face_data,rows[min_idx],cols[min_idx],A[min_idx]);

		_update_weights(best_face_data,best_non_face_data,face_weights,non_face_weights,theta,classifier_beta,classifier_polarity);

	}

}

void _update_weights(const vector<vector<float>>& face_data, const vector<vector<float>>& non_face_data, vector<float>& face_weights, vector<float>& non_face_weights, const float& theta, const float& classifier_beta, const int& classifier_polarity) {

	// update face weights
	for (int i = 0; i < face_data.size(); i++) {
		int classifier_output = 0;
		if (classifier_polarity == 1) {
			classifier_output = (face_data[0][i]-theta > 0) ? 1 : 0;
		} else {
			classifier_output = (theta-face_data[0][i] > 0) ? 1 : 0;
		}
		face_weights[i] = face_weights[i]*pow(classifier_beta,classifier_output);
	}

	// update non face weights
	for (int i = 0; i < non_face_data.size(); i++) {
		int classifier_output = 0;
		if (classifier_polarity == 1) {
			classifier_output = (face_data[0][i]-theta > 0) ? 1 : 0;
		} else {
			classifier_output = (theta-face_data[0][i] > 0) ? 1 : 0;
		}
		non_face_weights[i] = non_face_weights[i]*pow(classifier_beta,classifier_output);
	}
}

void _normalize_weights(vector<float>& weights) {

	float sum = 0;
	for (int i = 0; i < weights.size(); i++) {
		sum += weights[i];
	}
	for (int i = 0; i < weights.size(); i++) {
		weights[i] /= sum;
	}
}
int _find_min_idx(const vector<float>& error) {

	int minIdx = 0;
	float minError = error[0];
	for (int i = 1; i < error.size(); i++) {
		if (error[i] < minError) {
			minError = error[i];
			minIdx = i;
		}
	}
	return minIdx;

}

