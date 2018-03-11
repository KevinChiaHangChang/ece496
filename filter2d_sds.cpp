/******************************************************************************
 *
 * (c) Copyright 2012-2016 Xilinx, Inc. All rights reserved.
 *
 * This file contains confidential and proprietary information of Xilinx, Inc.
 * and is protected under U.S. and international copyright and other
 * intellectual property laws.
 *
 * DISCLAIMER
 * This disclaimer is not a license and does not grant any rights to the
 * materials distributed herewith. Except as otherwise provided in a valid
 * license issued to you by Xilinx, and to the maximum extent permitted by
 * applicable law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL
 * FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS,
 * IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
 * MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE;
 * and (2) Xilinx shall not be liable (whether in contract or tort, including
 * negligence, or under any other theory of liability) for any loss or damage
 * of any kind or nature related to, arising under or in connection with these
 * materials, including for any direct, or any indirect, special, incidental,
 * or consequential loss or damage (including loss of data, profits, goodwill,
 * or any type of loss or damage suffered as a result of any action brought by
 * a third party) even if such damage or loss was reasonably foreseeable or
 * Xilinx had been advised of the possibility of the same.
 *
 * CRITICAL APPLICATIONS
 * Xilinx products are not designed or intended to be fail-safe, or for use in
 * any application requiring fail-safe performance, such as life-support or
 * safety devices or systems, Class III medical devices, nuclear facilities,
 * applications related to the deployment of airbags, or any other applications
 * that could lead to death, personal injury, or severe property or
 * environmental damage (individually and collectively, "Critical
 * Applications"). Customer assumes the sole risk and liability of any use of
 * Xilinx products in Critical Applications, subject only to applicable laws
 * and regulations governing limitations on product liability.
 *
 * THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE
 * AT ALL TIMES.
 *
 *******************************************************************************/

#include <common/xf_common.h>
#include <imgproc/xf_custom_convolution.hpp>
#include <linux/videodev2.h>
#include <stdlib.h>

#include "filter2d_sds.h"

#define F2D_HEIGHT	2160
#define F2D_WIDTH	3840

using namespace xf;

struct filter2d_data {
	xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> *inLuma;
	xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> *inoutUV;
	xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> *outLuma;
	uint32_t in_fourcc;
	uint32_t out_fourcc;
};

#pragma SDS data copy("inoutUV.data"[0:"inoutUV.size"])
#pragma SDS data access_pattern("inoutUV.data":SEQUENTIAL)
#pragma SDS data mem_attribute("inoutUV.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(frm_data_in:NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data copy(frm_data_in[0:pcnt])
#pragma SDS data access_pattern(frm_data_in:SEQUENTIAL)
#pragma SDS data mem_attribute("inLuma.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data copy("inLuma.data"[0:"inLuma.size"])
#pragma SDS data access_pattern("inLuma.data":SEQUENTIAL)
void read_f2d_input(unsigned short *frm_data_in,
		    xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> &inLuma,
		    xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> &inoutUV,
		    uint32_t in_fourcc, int pcnt)
{
	unsigned short lumamask    = (V4L2_PIX_FMT_YUYV==in_fourcc)? 0x00FF : 0xFF00;
	unsigned short lumashift   = (V4L2_PIX_FMT_YUYV==in_fourcc)? 0      : 8;
	unsigned short chromamask  = (V4L2_PIX_FMT_YUYV==in_fourcc)? 0xFF00 : 0x00FF;
	unsigned short chromashift = (V4L2_PIX_FMT_YUYV==in_fourcc)? 8      : 0;

	for(int i=0; i<pcnt; i++){
#pragma HLS pipeline II=1
		unsigned short yuvpix = frm_data_in[i];
		ap_uint<8> ypix =  (ap_uint<8>)((yuvpix & lumamask)>>lumashift);
		ap_uint<8> uvpix = (ap_uint<8>)((yuvpix & chromamask)>>chromashift);
		inLuma.data[i] = ypix;
		inoutUV.data[i] = uvpix;
	}
}

#pragma SDS data buffer_depth("inoutUV.data":32768)
#pragma SDS data mem_attribute("inoutUV.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data copy("inoutUV.data"[0:"inoutUV.size"])
#pragma SDS data access_pattern("inoutUV.data":SEQUENTIAL)
#pragma SDS data mem_attribute("outLuma.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data copy("outLuma.data"[0:"outLuma.size"])
#pragma SDS data access_pattern("outLuma.data":SEQUENTIAL)
#pragma SDS data mem_attribute(frm_data_out:NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data copy(frm_data_out[0:pcnt])
#pragma SDS data access_pattern(frm_data_out:SEQUENTIAL)
void write_f2d_output(xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> &outLuma,
		      xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1> &inoutUV,
		      unsigned short *frm_data_out, uint32_t out_fourcc, int pcnt)
{
	unsigned short lumashift = (V4L2_PIX_FMT_YUYV==out_fourcc)? 0 : 8;
	unsigned short chromashift = (V4L2_PIX_FMT_YUYV==out_fourcc)? 8 : 0;

	for(int i=0; i<pcnt; i++){
#pragma HLS pipeline II=1
		ap_uint<8> ypix = outLuma.data[i];
		ap_uint<8> uvpix = inoutUV.data[i];
		unsigned short yuvpix = ((unsigned short) uvpix << chromashift) | ((unsigned short) ypix << lumashift);
		frm_data_out[i] = yuvpix;
	}
}

int filter2d_init_sds(size_t in_height, size_t in_width, size_t out_height,
		      size_t out_width, uint32_t in_fourcc,
		      uint32_t out_fourcc, void **priv)
{
	struct filter2d_data *f2d = (struct filter2d_data *) malloc(sizeof(struct filter2d_data));
	if (f2d == NULL) {
		return -1;
	}

	f2d->inLuma = new xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1>(in_height, in_width);
	f2d->inoutUV = new xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1>(in_height, in_width);
	f2d->outLuma = new xf::Mat<XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1>(in_height, in_width);
	f2d->in_fourcc = in_fourcc;
	f2d->out_fourcc = out_fourcc;

	*priv = f2d;

	return 0;
}

void filter2d_sds(unsigned short *frm_data_in, unsigned short *frm_data_out,
		  int height, int width, const coeff_t coeff, void *priv)
{
	struct filter2d_data *f2d = (struct filter2d_data *) priv;
	int pcnt = height*width;


	// split the 16b YUYV... input data into separate 8b YYYY... and 8b UVUV...
	read_f2d_input(frm_data_in, *f2d->inLuma, *f2d->inoutUV, f2d->in_fourcc, pcnt);

	// this is the xfopencv version of filter2D, found in imgproc/xf_custom_convolution.hpp
	xf::filter2D<XF_BORDER_CONSTANT, KSIZE, KSIZE, XF_8UC1, XF_8UC1, F2D_HEIGHT, F2D_WIDTH, XF_NPPC1>
		(*f2d->inLuma, *f2d->outLuma, (short int *) coeff, 0);

	// combine separate 8b YYYY... and 8b UVUV... data into 16b YUYV... output data
	write_f2d_output(*f2d->outLuma, *f2d->inoutUV, frm_data_out, f2d->out_fourcc, pcnt);
}

#define NUM_FACES			200
#define NUM_NON_FACES		400
#define NUM_HAAR_FEATURES	10

// weights for face images = 1/(2*NUM_FACES)
// weights for non-face images = 1/(2*NUM_NON_FACES)

// haar feature extraction matrices

xf::Mat A0 = (Mat_<float>(6,3) << -1,-1,-1,/**/-1,-1,-1,/**/-1,-1,-1,/**/1,1,1,/**/1,1,1,/**/1,1,1);
xf::Mat A1 = (Mat_<float>(6,6) << -1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,/**/1,1,1,1,1,1,/**/1,1,1,1,1,1,/**/1,1,1,1,1,1);
xf::Mat A2 = (Mat_<float>(6,9) << -1,-1,-1,-1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,-1,-1,-1,/**/-1,-1,-1,-1,-1,-1,-1,-1,-1,/**/1,1,1,1,1,1,1,1,1,/**/1,1,1,1,1,1,1,1,1,/**/1,1,1,1,1,1,1,1,1);
xf::Mat A3 = (Mat_<float>(8,4) << -1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1,/**/-1,-1,1,1);
xf::Mat A4 = (Mat_<float>(8,6) << -1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1,/**/-1,-1,-1,1,1,1);
xf::Mat A5 = (Mat_<float>(8,4) << -1,-1,-1,-1,/**/-1,-1,-1,-1,/**/-1,-1,-1,-1,/**/-1,-1,-1,-1,/**/-1,-1,-1,-1,/**/1,1,1,1,/**/1,1,1,1,/**/1,1,1,1,/**/1,1,1,1);
xf::Mat A6 = (Mat_<float>(8,4) << 1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1,/**/1,-1,-1,1);
xf::Mat A7 = (Mat_<float>(8,6) << 1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1,/**/1,1,-1,-1,1,1);
xf::Mat A8 = (Mat_<float>(8,7) << 1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1,/**/1,1,-1,-1,-1,1,1);
xf::Mat A9 = (Mat_<float>(8,9) << 1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1,/**/1,1,1,-1,-1,-1,1,1,1);
xf::Mat A[NUM_HAAR_FEATURES] = {A0,A1,A2,A3,A4,A5,A6,A7,A8,A9};

// weights
std::vector<float> face_weights(NUM_FACES,0);
std::vector<float> non_face_weights(NUM_NON_FACES,0);

void adaboost_loop() {

	// initialize face weights
	for (int i = 0; i < NUM_FACES; i++) {
		face_weights[i] = 1/(2*NUM_FACES);
	}
	// initialize non-face weights
	for (int i = 0; i < NUM_NON_FACES; i++) {
		non_face_weights[i] = 1/(2*_NUM_NON_FACES);
	}

	// main loop
	for (int i = 0; i < 3; i++) {

		// normalize face weights
		float sum = 0;
		for (int j = 0; j < NUM_FACES; j++) {
			sum += face_weights[j];
		}
		for (int j = 0; j < NUM_FACES; j++) {
			face_weights[j] /= sum;
		}
		// normalize non-face weights
		sum = 0;
		for (int k = 0; k < NUM_NON_FACES; k++) {
			sum += non_face_weights[k];
		}
		for (int k = 0; k < NUM_NON_FACES; k++) {
			non_face_weights /= sum;
		}

		// initialize error, beta, threshold vectors + polarity list
		std::vector<float> error(NUM_HAAR_FEATURES,0);
		std::vector<float> beta(NUM_HAAR_FEATURE,0);
		std::vector<float> threshold(NUM_HAAR_FEATURES,0);
		std::vector<int> polarity(NUM_HAAR_FEATURES,0);


		// select best weak classifier w/ respect to current weights
		for (int l = 0; l < NUM_HAAR_FEATURES; l++) {

			// extract data according to Haar feature
			// TODO
			xf::Mat face_data;
			xf::Mat non_face_data;

			// perform PLA
			// TODO
			float currError;
			float currBeta = currError/(1-currError);
			float currThreshold;
			int currPolarity;

			// update error
			error[l] = currError;
			beta[l] = currBeta;
			threshold[l] = currThreshold;
			polarity[l] = currPolarity;

		}

		// define classifier
		// find index of min
		float minError = error[0];
		int minIdx = 0;
		for (int m = 1; m < NUM_HAAR_FEATURES; m++) {
			if (error[m] < minError) {
				minError = error[m];
				minIdx = m;
			}
		}
		float theta = threshold[minIdx];
		xf::Mat bestFeature = A[minIdx];
		int classifierPolarity = polarity[minIdx];
		float classifierBeta = beta[minIdx];
		float alpha = log(1/classifierBeta);

		// update weights of misclassified points
		for (int n = 0; n < NUM_FACES; n++) {
			int classifierOutput = 0;
			if (classifierPolarity == 1) {
				classifierOutput = face_data[n]-theta;
				classifierOutput = (classifierOutput > 0) ? 1 : 0;
			} else {
				classifierOutput = theta-face_data[n];
				classifierOutput = (classifierOutput > 0) ? 1 : 0;
			}
			face_weights[n] = face_weights[n]*pow(classifierBeta,classifierOutput);
		}

	}
}




/*void violajones_sds(unsigned short *frm_data_in, unsigned short *frm_data_out,
			int height, int width, const coeff_t coeff, void *priv)
{
	// filter2d data
	struct filter2d_data *f2d = (struct filter2d_data *) priv;
	// pixel count
	int pcnt = height*width;

	// split 16b YUYV... input data into separate 8b YYYY... and 8b UVUV...
	read_f2d_input(frm_data_in, *f2d->inLuma, *f2d->inoutUV, f2d->in_fourcc, pcnt);

	// integral image
	// template<int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC=1>
	// xf::integral();
	xf::integral(*f2d->inLuma, *f2d->outLuma);

}

void findoptimal_sds(std::vector<int> &y_target, std::vector<float> &weights, float &min_error, int &polarity)
{
	// initialize error + polarity list
	vector<int> error_list(y_target.size(), -1);
	vector<int> polarity_list(y_target.size(), 0);

	float T_plus = 0;
	float T_minus = 0;
	// count number of positive/negative samples
	// find total sum of positive sample weights T_plus
	// find total sum of negative sample weights T_minus
	for (int i = 0; i < y_target.size(); ++i) {
		if (y_target[i] == 1) {
			total_pos++;
			T_plus += weights[i];
		} else if (y_target[i] == -1) {
			total_neg++;
			T_minus += weights[i];
		}
	}

	// iterate over data
	for (int i = 0; i < y_target.size(); ++i) {
		// find sum of positive weights below current sample S_plus
		// find sum of negative weights below current sample S_minus
		float S_plus = 0;
		float S_minus = 0;
		for (int j = 0; j < y_target.size(); ++j) {
			if (y_target[j] == 1 && weights[j] < weights[i]) {
				S_plus += weights[j];
			} else if (y_target[j] == -1 && weights[j] < weights[i]) {
				S_minus += weights[j];
			}
		}
		// update error + polarity list
		if (S_plus+(T_minus-S_minus) < S_minus+(T_plus-S_plus)) {
			error_list[i] = S_plus+(T_minus-S_minus);
			polarity[i] = 1;
		} else {
			error_list[i] = S_minus+(T_plus-S_plus);
			polarity[i] = -1;
		}
	}

	// find minimum error
	int min_error_index = 0;
	for (int i = 0; i < y_target.size(); ++i) {
		if (error_list[i] < error_list[min_error_index]) {
			min_error_index = i;
		}
	}
	min_error = error_list[min_error_index];
	polarity = polarity_list[min_error_index];
	// return best x

}*/

// weights for face images = NUM_FACES/(2*NUM_FACES)
// weights for non-face images = NUM_NON_FACES/(2*NUM_NON_FACES)

/*void adaboost_sds()
{
	// initialize weights
	std::vector<float> weights;
	for (int i = 0; i < NUM_FACES+NUM_NON_FACES; ++i) {
		if (i < NUM_FACES) {
			weights.push_back(1/(2*NUM_FACES));
		} else {
			weights.push_back(1/(2*NUM_NON_FACES));
		}
	}

	// initialize error list
	std::vector<float> error_list(NUM_HAAR_FEATURES, 0.0);

	// initialize beta list
	std::vector<float> beta_list(NUM_HAAR_FEATURES, 0.0);

	// initialize threshold list
	std::vector<float> threshold_list(NUM_HAAR_FEATURES, 0.0);

	// initialize polarity list
	std::vector<int> polarity_list(NUM_HAAR_FEATURES, 0.0);

	// main loop
	// three stage cascade classifier
	for (int i = 0; i < 3; ++i) {
		// normalize weights
		float sum = std::accumulate(weights.begin(), weights.end(), 0.0);
		for (int j = 0; j < weights.size(); ++j) {
			weights[j] = weights[j]/sum;
		}

		// select best weak classifier w/ respect to weighted error
		float min_error = 0;
		int polarity = 0;
		// findoptimal_sds(y_target, weights, min_error, polarity);
		beta = min_error/(1-min_error);

	}
}*/

