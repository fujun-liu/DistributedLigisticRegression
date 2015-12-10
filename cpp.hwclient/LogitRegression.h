#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <zmq.hpp>
#include "zmqClient.h"


class LogitRegression
{
private:
	float *feat;
	int *label;
	int num, featDim;
	int epoches, batchsz;
	float learning_rate;
	float *w, *derv;
	int pull_stride, push_stride;
	zmqClient* pClient;
	// for output
	std::string prefix;
public:
	LogitRegression(std::string feat_path, std::string label_path, std::string server_port = "");
	
	void Init();
	void set_pull_stride(int pull_stride){ this->pull_stride = pull_stride;};
	void set_push_stride(int push_stride){ this->push_stride = push_stride;};
	void set_learning_rate(float learning_rate){this->learning_rate = learning_rate;}
	void set_batchsz(int batchsz){this->batchsz = batchsz;}
	void set_epoches(int epoches) {this->epoches = epoches;};

	void train();
	void train_parameter_server();
	// pull and push 
	void pull_weight();
	void push_derv();
	
	// knuth shuffling 
	void shuffling_knuth();
	// write
	void write_weights(std::string);
	//
	float compute_accuracy(const float* x, const int* y, int num);

	float approxLogExp(float x);
	~LogitRegression(void);
};

