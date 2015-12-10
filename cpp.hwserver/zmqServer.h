#pragma once

#include <zmq.hpp>
#include <string>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>

#define sleep(n)	Sleep(n)
#endif
#include <limits>

#include <time.h>
#include "zmqprotobuf.pb.h"
#include "paras.h"
#include "ps_constants.h"


class zmqServer
{
private:
	zmq::context_t *context;
	zmq::socket_t *socket;
	
	int num_weight;
	float *weight;
	float *deriv_prev;

	float *deriv_square_sum; // used for Adagrad to adjust learning rate

	float alpha; // learning rate
	float momentum; // momentum
	float gamma, power;
	bool use_momentum;

	int learning_policy; // 0 for 

	zmqprotobuf::PSMSG rep_weight; // preallocated, save allocting time with extra space
	long iter;
public:
	// constructor
	zmqServer(std::string server_port, std::string weight_init_path = "");

	void parseConfigureFile(std::string configure_file_path);

	void load_weight(const std::string weight_path); // init weight, weights initlization will be suported later 

	void update_weight(zmqprotobuf::PSMSG &pmsg);

	void update_weight_basic(zmqprotobuf::PSMSG &pmsg); // no momentum
	void update_weight_adagrad(zmqprotobuf::PSMSG &pmsg); // adagrad algorithm
	void update_weight_momentum(zmqprotobuf::PSMSG &pmsg); // using momentum
	
	float learning_rate(); // check current learning rate based on learning policy
	void save_weight();

	void run(); // handling client's request
	
	~zmqServer(void);
};

