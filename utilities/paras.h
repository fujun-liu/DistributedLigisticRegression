#ifndef PARAS_H
#define PARAS_H

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "ps_constants.h"

class Paras{
private:
	int node_type; // server or client
	// for server
	int learning_policy; // 

	double alpha; // learning rate
	double momentum; // momentum
	double gamma, power; // for inv learning policy

	int nIters;
	int weightNum; // dimension

	// for client
	std::string feat_path, label_path; // training data for client
	std::string server_port; // tell the client the server port
	int batchsz; // batch size for training
	int epoches; // number of epoches
public:
	Paras(std::string config_path);
	int get_learning_policy() { return learning_policy;}
	double get_learning_rate() { return alpha;}
	double get_momentum() { return momentum;}
	double get_gamma() { return gamma;}
	double get_power() { return power;}
	double get_weightNum() { return weightNum;}
	std::string get_feat_path() {return feat_path; }
	std::string get_label_path() {return label_path; }
	std::string get_server_port() { return server_port; }
	int get_batch_size() {return batchsz; }
	int get_epoches() {return epoches; }

	void init();
	
	void parseNodeConfig(std::string key, std::string value, int node_type);

	void displayServerInfo();
	~Paras() {};
};
#endif