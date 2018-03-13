#ifndef __COMMON_H__
#define __COMMON_H__


typedef struct opt {

	float bestx;
	float min_error;
	int polarity;
} opt;

typedef struct data {

	unsigned short avg;
	int target;
	int polarity;
} data;

#endif /* __COMMON_H__ */
