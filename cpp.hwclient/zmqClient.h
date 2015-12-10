#pragma once

#include <zmq.hpp>
#include <string>
#include <iostream>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#define sleep(n)	Sleep(n)
#endif

#include "zmqprotobuf.pb.h"
#include "ps_constants.h"

class zmqClient
{
private:
	zmq::context_t *context;
	zmq::socket_t *socket;
public:
	zmqClient(std::string server);
	
	void zmqGet(zmqprotobuf::PSMSG & msg);
	void zmqGet(float* weight, int num);

	void zmqPut(zmqprotobuf::PSMSG & msg);
	void zmqPut(float* derv, int num);
	~zmqClient();
};

