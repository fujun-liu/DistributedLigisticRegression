#include <iostream>
#include "logitRegression.h"
#include <math.h>

int main(){
	std::string feat_path = "..\\data\\trainx.bin", label_path = "..\\data\\trainy.bin";
	LogitRegression model(feat_path, label_path);
	model.set_learning_rate(0.01);
	model.set_batchsz(100);
	model.set_epoches(100);

	model.train();
	//cout << "accuracy is"
	std::cout << std::endl << "Press Enter key to exit ...";
	std::cin.get();
	return 0;
}