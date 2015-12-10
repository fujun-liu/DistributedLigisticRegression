#include <iostream>
#include "logitRegression.h"
#include <math.h>
#include "paras.h"

int main(int argc, char ** argv){
	if (argc != 2){
		std::cout << "Usage: run_client.exe config_path" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::string configure_file_path(argv[1]);
	Paras* paras = new Paras(configure_file_path);

	LogitRegression model(paras->get_feat_path(), paras->get_label_path(), paras->get_server_port());

	//model.set_learning_rate(0.01);
	model.set_batchsz(paras->get_batch_size());
	model.set_epoches(paras->get_epoches());
	//model.train();
	
	model.train_parameter_server();

	//cout << "accuracy is"
	std::cout << std::endl << "Press Enter key to exit ...";
	std::cin.get();
	return 0;
}

