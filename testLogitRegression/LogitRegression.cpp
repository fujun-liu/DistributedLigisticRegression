#include "LogitRegression.h"


LogitRegression::LogitRegression(std::string feat_path, std::string label_path){
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
	//std::fill_n(w, featDim+1, 0);
	srand(time(NULL));
	float rand_max_val = RAND_MAX;
	//std::cout << "initial weights are:" << std::endl;
	for (int i = 0; i <= featDim; ++ i){
		w[i] = rand() / rand_max_val;
		//std::cout << w[i] << " ";
	}
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

	return;
}

// push_derv will upload derv to server
void LogitRegression::push_derv(){
	return;
}


void LogitRegression::train_parameter_server(){
	int batches = num/batchsz;
	assert(batches > 0);
	long iter = 1; 
	for (int epoch = 0; epoch < epoches; ++ epoch){
		// handle one batch in each batch_idation
		for (int batch_id = 0; batch_id < batches; ++ batch_id){
			if(iter%pull_stride == 0){
				pull_weight();
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

			std::cout << "loss is " << loss/batchsz << std::endl;
			for (int j = 0; j <= featDim; ++ j){
				derv[j] /= batchsz;
			}

			if (iter%push_stride == 0){
				push_derv();
			}

			// update weights
			for (int j = 0; j <= featDim; ++ j){
				w[j] += -learning_rate * derv[j];	
			}

			++ iter;
		}
	}

	std::cout << "Traning accuracy is " << accuracy(feat, label, num) << std::endl;


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

	std::cout << "Traning accuracy is " << accuracy(feat, label, num) << std::endl;
}


float LogitRegression::accuracy(const float* x, const int* y, int num){
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
