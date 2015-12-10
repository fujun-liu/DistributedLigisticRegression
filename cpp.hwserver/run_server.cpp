#include <string>
#include <iostream>

#include "zmqServer.h"
#include "paras.h"


int main(int argc, char** argv){
	if (argc != 3){
		std::cout << "Usage: run_server.exe server_port[tcp://x.x.x.x:yyyy] config_path[configuration file path]" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	// load initial weight and learning paras
	std::string server_port(argv[1]);
	std::string config_path(argv[2]);

	//std::string server_port = "tcp://*:5555";
	//std::string config_path = "..\\data\\configure.txt";

	zmqServer* ps = new zmqServer(server_port, config_path);
	ps->run();

	system("pause");
	return 0;
}