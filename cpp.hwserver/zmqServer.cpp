#include "zmqServer.h"
#include <math.h>
#include <fstream>
#include <iostream>

zmqServer::zmqServer(std::string server_port, std::string configure_path)
{
	context = new zmq::context_t(1);
	socket = new zmq::socket_t (*context, ZMQ_REP);
	//ready to receive request
	socket->bind(server_port.c_str());

	parseConfigureFile(configure_path);
	

	// allocate memeory for reply
	rep_weight.set_type(PUSH_REQUEST);
	rep_weight.mutable_weights()->Reserve(num_weight);
	for (int i = 0; i < num_weight; ++ i){
		rep_weight.add_weights(0);
	}


	weight = new float[num_weight];
	// init weight randomly
	srand(time(NULL));
	for (int i = 0; i < num_weight; ++ i){
		weight[i] = rand()/(float)RAND_MAX;
	}

	if (learning_policy == LEARN_POLICY_ADAGRAD){
		this->deriv_square_sum = new float[num_weight];
		std::fill_n(deriv_square_sum, num_weight, 0);
	}

	use_momentum = momentum > FLOAT_0;

	if (use_momentum){
		deriv_prev = new float[num_weight];
	}

}

float zmqServer::learning_rate(){
	float curr = alpha;
	switch(learning_policy){
		case LEARN_POLICY_INV:
			curr = alpha*pow(1+gamma*iter, -power);
			break;
	}
	return curr;
}

void zmqServer::parseConfigureFile(std::string configure_file_path){
	Paras* paras = new Paras(configure_file_path);
	paras->displayServerInfo();

	this->learning_policy = paras->get_learning_policy();
	this->alpha = paras->get_learning_rate();
	this->gamma = paras->get_gamma();
	this->power = paras->get_power();

	this->num_weight = paras->get_weightNum();
}

// no momentum used
void zmqServer::update_weight_basic(zmqprotobuf::PSMSG &pmsg){
	float lr = learning_rate();
	for (int i = 0; i < num_weight; ++ i){
		weight[i] += -lr*pmsg.gradients(i);
	}
}


void zmqServer::update_weight_adagrad(zmqprotobuf::PSMSG &pmsg){
	for (int i = 0; i < this->num_weight; ++ i){
		this->deriv_square_sum[i] += pow(pmsg.gradients(i), 2.0);
		// Adagrad
		this->weight[i] += -alpha*pmsg.gradients(i)/sqrt(this->deriv_square_sum[i]);
	}
}

void zmqServer::update_weight_momentum(zmqprotobuf::PSMSG &pmsg){
	float lr = learning_rate();
	for (int i = 0; i < num_weight; ++ i){
		float v = -lr*pmsg.gradients(i);
		v += momentum * deriv_prev[i]; // use momentum

		weight[i] += v;
		deriv_prev[i] = v;
	}
}


void zmqServer::update_weight(zmqprotobuf::PSMSG &pmsg){
	if (num_weight != pmsg.gradients_size()){
		std::cout << "weight size (" << num_weight << ") and gradient sizes (" << pmsg.gradients_size() << ") don't match!" << std::endl;
		return;
	}

	if (learning_policy == LEARN_POLICY_ADAGRAD){// momentum will be ignored in this case
		update_weight_adagrad(pmsg);
	} else if (use_momentum){
		update_weight_momentum(pmsg);
	}else{ // basic learning rate
		update_weight_basic(pmsg);
	}
	
}

void zmqServer::load_weight(const std::string weight_path){
	std::ifstream fd(weight_path.c_str(), std::ios::in | std::ios::binary);
	fd.read((char*)&num_weight, 4);
	weight = new float[num_weight];
	fd.read((char*) weight, 4*num_weight);
	fd.close();
	
	// check the weights are loaded correctly

	/*int test_ids[] = {0, 100, 200, 300, num_weight-1};
	std::cout << "There are totall " << num_weight << " weights, and sample values are:" << std::endl;
	for (int i = 0; i < sizeof(test_ids)/sizeof(test_ids[0]); ++ i){
		std::cout << test_ids[i] << ": " << weight[test_ids[i]] << std::endl;
	}*/

}
void zmqServer::run(){
	  while (true) {
		std::cout << "waiting for request ..." << std::endl;
        zmq::message_t request;
        //  Wait for next request from client
        socket->recv (&request);
		// get the request and read them into string
		std::string req_data = std::string(static_cast<char*>(request.data()), request.size());

		// parse the request data into protobuf data structure
		zmqprotobuf::PSMSG pmsg;
		pmsg.ParseFromString(req_data);
		
		if (pmsg.type() == PULL_REQUEST){ // client asking for weights
			//  package current weight into protobuf struture 
			memcpy(rep_weight.mutable_weights()->mutable_data(), weight, sizeof(float)*num_weight);
			std::string rep_data = rep_weight.SerializeAsString();

			// send reply back to client
			zmq::message_t reply (rep_data.length());
			memcpy ((void *) reply.data (), rep_data.data(), rep_data.length());
			socket->send (reply);

		} else if (pmsg.type() == PUSH_REQUEST){
			// update gradients
			update_weight(pmsg);
			++ iter;

			// reply to client anyway, not efficient, requreied by the socket type
			std::string rep_msg = "weights updated"; 
			zmq::message_t request (rep_msg.length());
			memcpy ((void *) request.data (), rep_msg.data(), rep_msg.length());
			// send the message through socket
			socket->send (request);
		}else{
			std::cout << "unknown requirement!" << std::endl;
		}

		//system("pause");
    }
}


zmqServer::~zmqServer(void)
{
	socket->close();
	context->close();

	delete []weight;

	if (this->learning_policy == LEARN_POLICY_ADAGRAD){
		delete []deriv_square_sum;
	}

	if (momentum > FLOAT_0){
		delete [] deriv_prev;
	}
}
