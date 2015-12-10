#include "paras.h"

void Paras::init(){
	node_type = -1;
	// for server
	learning_policy = 0;
	alpha = 1.0;
	momentum = 0.;
	gamma = 0.0001;
	power = 0.75;
	weightNum = 0;

	// for client
	feat_path = "";
	label_path = "";
	server_port = "";
	batchsz = 32;
	epoches = 1;
}

void Paras::displayServerInfo(){
	std::cout << "The configuration information: " << std::endl;
	std::cout << std::setw(15) << std::left << "alpha: " << alpha << std::endl;
	std::cout << std::setw(15) << std::left << "momentum: " << momentum << std::endl;
	std::cout << std::setw(15) << std::left << "gamma: " << gamma << std::endl;
	std::cout << std::setw(15) << std::left << "power: " << power << std::endl;
	std::cout << std::setw(15) << std::left << "weightNum: " << weightNum << std::endl;
}

void Paras::parseNodeConfig(std::string key, std::string value, int node_type){
	// check key
	if (node_type == SERVER_NODE_TYPE){ // for server
		if (key.compare("alpha") == 0){
			alpha = atof(value.c_str());
		}else if (key.compare("momentum") == 0){
			momentum = atof(value.c_str());
		}else if (key.compare("gamma") == 0){
			gamma = atof(value.c_str());
		}else if (key.compare("power") == 0){
			power = atof(value.c_str());
		}else if(key.compare("weight_num") == 0){
			weightNum = atoi(value.c_str());
		}else if(key.compare("niters") == 0){
			nIters = atoi(value.c_str());
		}else if(key.compare("learning_policy") == 0){
			learning_policy = atoi(value.c_str());
		}
	}else if (node_type == CLIENT_NODE_TYPE){// for 
		if (key.compare("feat_path") == 0){
			feat_path = value;
		}else if(key.compare("label_path") == 0){
			label_path = value;
		}else if(key.compare("server_port") == 0){
			server_port = value;
		}else if(key.compare("batch_size") == 0){
			batchsz = atoi(value.c_str());
		}else if(key.compare("epoches") == 0){
			epoches = atoi(value.c_str());
		}
	}
	
}

Paras::Paras(std::string config_path){
	// default parameters
	init();

	std::ifstream ifs(config_path);
	if(!ifs.is_open()){
		std::cout << "configuration path " << config_path << " is not set right." << std::endl;
	}

	std::string line;
	std::cout << "parsing " << config_path << "..." << std::endl;
	while(std::getline(ifs, line)){
		std::cout << line << std::endl;

		// remove space in line
		line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
		if (line.empty() || line[0] == '#'){// ignore this line
			continue;
		}

		std::size_t found = line.find('=');
		if (found != std::string::npos){
			std::string key = line.substr(0, found);
			std::string value = line.substr(found+1);
			// convert key and value to lower
			std::transform(key.begin(), key.end(), key.begin(), ::tolower);
			std::transform(value.begin(), value.end(), value.begin(), ::tolower);

			// the first expected is nodeType, server or client
			if (key.compare("node_type") == 0){
				if (value.compare("server") == 0){
					node_type = SERVER_NODE_TYPE;
				}else if (value.compare("client") == 0){
					node_type = CLIENT_NODE_TYPE;
				}
			}else if(node_type == SERVER_NODE_TYPE || node_type == CLIENT_NODE_TYPE){
				parseNodeConfig(key, value, node_type);
			}else{
				std::cout << "node type (client or server) is not specified" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			
		}

	}
	
	if (node_type == SERVER_NODE_TYPE && weightNum == 0){
		std::cout << "weignt Num is not set. This is required now." << std::endl;
		std::exit(EXIT_FAILURE);
	}else if (node_type == CLIENT_NODE_TYPE && (
		feat_path.empty() || label_path.empty() || server_port.empty())){
		std::cout << "feat_path, label_path or server_port is not set. This is required now." << std::endl;
		std::exit(EXIT_FAILURE);
	}else{
		std::cout << "parsing is finished succesfully!" << "\n\n";
	}
}

