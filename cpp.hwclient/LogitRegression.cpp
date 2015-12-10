#include "LogitRegression.h"
#include <vector> // vector
#include <utility> // pair

LogitRegression::LogitRegression(std::string feat_path, std::string label_path, std::string server_port){
	if (server_port.empty()){
		pClient = NULL;
	}else{
		pClient = new zmqClient(server_port);
	}
	// read feature
	{
		std::ifstream fid(feat_path.c_str(), std::ios::in | std::ios::binary);
		if (!fid.is_open()){
			std::cout << "please check file path: " << feat_path << std::endl;
			return;
		}
		// read num and featDim
		fid.read((char *)&num, 4);
		fid.read((char *)&featDim, 4);
		std::cout << "# of samples: " << num << ", feature dim: " << featDim << std::endl;

		// read data
		feat = new float[num*featDim];
		fid.read((char *)feat, num*featDim*4);
		fid.close();

		/*std::cout << "Last data is: " << std::endl;
		for (int j = 0; j < featDim; ++ j){
			std::cout << feat[(num-1)*featDim + j] << " ";
		}
		std::cout << std::endl;*/
	}

	// read label
	{
		std::ifstream fid(label_path, std::ios::in | std::ios::binary);
		if (!fid.is_open()){
			std::cout << "please check file path: " << label_path << std::endl;
			return;
		}

		label = new int[num];
		int num2;
		fid.read((char *)&num2, 4);
		std::cout << "# of labels: " << num2 << std::endl;

		assert(num2 == num); 
		fid.read((char *)label, num*4);
		fid.close();

		// convert label to {-1, 1}. Now label is {0, 1}.
		for (int i = 0; i < num; ++ i){
			
			label[i] = 2*label[i] - 1;
			
			//std::cout << "label of " << i << " is " << label[i] << std::endl;
		}

		Init();
	}

	// weights are initialized as 0s
	w = new float[featDim + 1];
	std::fill_n(w, featDim+1, 0);
	
	/*srand(time(NULL));
	float rand_max_val = RAND_MAX;
	//std::cout << "initial weights are:" << std::endl;
	for (int i = 0; i <= featDim; ++ i){
		w[i] = rand() / rand_max_val;
		//std::cout << w[i] << " ";
	}*/
	//std::cout << std::endl;

	derv = new float[featDim + 1];
	std::fill_n(derv, featDim + 1, 0);
	
}

void LogitRegression::Init(){
		this->epoches = 1000;
		this->batchsz = 10;
		this->learning_rate = 1;
		this->pull_stride = 1;
		this->push_stride = 1;

		// init prefix as time
		time_t rawtime;
		struct tm* timeinfo;
		time(&rawtime); // get raw tile
		timeinfo = localtime(&rawtime);
		char time_buffer[80];
		strftime(time_buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
		this->prefix = std::string(time_buffer);
}

float LogitRegression::approxLogExp(float x){
	float ret = 0;
	if (x < -10)
		ret = exp(x);
	else if( x > 35)
		ret = x;
	else{
		ret = log(1 + exp(x));
	}

	return ret;
}

// pull_weight will fetch weight from server
void LogitRegression::pull_weight(){
	pClient->zmqGet(w, featDim+1);
}

// push_derv will upload derv to server
void LogitRegression::push_derv(){
	pClient->zmqPut(derv, featDim+1);
}

void LogitRegression::write_weights(std::string outfile_name){
	std::ofstream outfile;
	outfile.open(outfile_name, std::ios::out | std::ios::trunc);
	if (outfile.is_open()){
		for (int i = 0; i <= featDim; ++ i){
			outfile << w[i] << " ";
		}
		outfile << std::endl;
		outfile.close();
	}else{
		std::cout << "Target file " << outfile_name << " was not open, write failed" << std::endl;
	}
	
}
void LogitRegression::train_parameter_server(){
	int batches = num/batchsz;
	assert(batches > 0);
	long iter = 1;
	std::vector<std::pair<long, float>> stat;
	for (int epoch = 0; epoch < epoches; ++ epoch){
		// shuffle whole dataset using knuth algorithm
		shuffling_knuth();

		// handle one batch in each batch_idation
		for (int batch_id = 0; batch_id < batches; ++ batch_id){
			
			if(iter%pull_stride == 0){
				pull_weight();
				if (iter == 1){
					float acc = compute_accuracy(feat, label, num);
					stat.push_back(std::make_pair(iter, acc));

					write_weights(prefix + "_weights_init.txt");
					std::cout << "Iter " << iter << ", traning accuracy is " << acc << std::endl;
				}
			}

			if (iter % 2 == 0){
				float acc = compute_accuracy(feat, label, num);
				stat.push_back(std::make_pair(iter, acc));

				std::cout << "Iter " << iter << ", traning accuracy is " << acc << std::endl;
			}
			std::fill_n(derv, featDim+1, 0);
			int batch_volume = batchsz*featDim;
			// compute gradient
			float loss = 0;
			for (int i = 0; i < batchsz; ++ i){
				// compute z
				float z = w[0], z1;
				for (int j = 0; j < featDim; ++ j){
					z += w[j+1] * feat[batch_id*batch_volume + i*featDim + j];
				}
				
				loss += approxLogExp(-label[batch_id*batchsz + i] * z);

				z1 = 1.0 - 1.0/(1.0 + exp(-label[batch_id*batchsz + i]*z));
				for (int j = 0; j <= featDim; ++ j){
					
					if (j == 0){
						derv[j] += -label[batch_id*batchsz + i] * z1;
					}else{
						derv[j] += -label[batch_id*batchsz + i] * z1 * feat[batch_id*batch_volume + i*featDim + j - 1];
					}

				}
			}

			//std::cout << "loss is " << loss/batchsz << std::endl;
			for (int j = 0; j <= featDim; ++ j){
				derv[j] /= batchsz;
			}

			if (iter%push_stride == 0){
				push_derv();
			}

			++ iter;
		}
		

	}

	write_weights(prefix + "_weights_optmized.txt");
	float acc = compute_accuracy(feat, label, num);
	std::cout << "Traning accuracy is " << acc << std::endl;
	stat.push_back(std::make_pair(iter, acc));
	// write accuracy results to file
	std::string stat_fn = prefix + "_accuracy.txt";

	std::fstream outfile;
	outfile.open(stat_fn, std::ios::out | std::ios::trunc);
	if (outfile.is_open()){
		for (auto p:stat){
			outfile << p.first << " " << p.second << std::endl;
		}
		outfile.close();

	}else{

		std::cout << "Target file: " << stat_fn << " was not opened, write failed." << std::endl;
	}
}


void LogitRegression::train(){
	int batches = num/batchsz;
	assert(batches > 0);
	for (int epoch = 0; epoch < epoches; ++ epoch){
		// handle one batch in each batch_idation
		for (int batch_id = 0; batch_id < batches; ++ batch_id){
			std::fill_n(derv, featDim+1, 0);
			int batch_volume = batchsz*featDim;
			// compute gradient
			float loss = 0;
			for (int i = 0; i < batchsz; ++ i){
				// compute z
				float z = w[0], z1;
				for (int j = 0; j < featDim; ++ j){
					z += w[j+1] * feat[batch_id*batch_volume + i*featDim + j];
				}
				
				loss += approxLogExp(-label[batch_id*batchsz + i] * z);

				z1 = 1.0 - 1.0/(1.0 + exp(-label[batch_id*batchsz + i]*z));
				for (int j = 0; j <= featDim; ++ j){
					
					if (j == 0){
						derv[j] += -label[batch_id*batchsz + i] * z1;
					}else{
						derv[j] += -label[batch_id*batchsz + i] * z1 * feat[batch_id*batch_volume + i*featDim + j - 1];
					}

				}
			}

			std::cout << "loss is " << loss/batchsz << std::endl;

			// update weights
			for (int j = 0; j <= featDim; ++ j){
				w[j] += -learning_rate * derv[j]/batchsz;	
			}
		}
	}

	std::cout << "Traning accuracy is " << compute_accuracy(feat, label, num) << std::endl;
}


void LogitRegression::shuffling_knuth(){
	if (feat == NULL || label == NULL){
		return;
	}

	// get distinct seed
	srand(time(NULL));
	float* feat_tmp = new float[featDim];
	size_t bytes = sizeof(float) * featDim;
	for (int i = 0; i < num; ++ i){
		int j = i + (int)(rand()/(1.0 + RAND_MAX) * (num-i)); // random num [i num-1]
		// swap label
		{
			int tmp = label[i];
			label[i] = label[j];
			label[j] = tmp;
		}
		// swap feat
		{
			memcpy(feat_tmp, feat + i*featDim, bytes);
			memcpy(feat + i*featDim, feat + j*featDim, bytes);
			memcpy(feat + j*featDim, feat_tmp, bytes);
		}
	}
	delete[] feat_tmp;
}
float LogitRegression::compute_accuracy(const float* x, const int* y, int num){
	float acc = .0;
	for (int i = 0; i < num; ++ i){
		float pred = w[0];
		for (int j = 0; j < featDim; ++ j){
			pred += w[j+1] * x[i*featDim + j];
		}
		
		if (y[i]*pred > 0) // label is {-1, +1}
			acc += 1.0;
	}
	return acc/num;
}

LogitRegression::~LogitRegression(void)
{
	delete[] feat; feat = NULL;
	delete[] label; label = NULL;
	delete[] w; w = NULL;
	delete[] derv; derv = NULL;
}
