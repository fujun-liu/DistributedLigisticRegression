#include "zmqClient.h"


zmqClient::zmqClient(std::string server)
{
	context = new zmq::context_t(1);
	socket = new zmq::socket_t (*context, ZMQ_REQ);
	socket->connect(server.c_str());
}

// 
void zmqClient::zmqGet(zmqprotobuf::PSMSG & pmsg){
	// send message to ask for weights
	std::string outmsg = pmsg.SerializeAsString();
	zmq::message_t request (outmsg.length());
	memcpy ((void *) request.data (), outmsg.data(), outmsg.length());
	// send the message through socket
	socket->send (request);

	// receive weights from server
	zmq::message_t reply;
	socket->recv (&reply);
	// put the data replied from server into string, then decode it
	std::string reply_msg = std::string(static_cast<char*>(reply.data()), reply.size());
	pmsg.ParseFromString(reply_msg);
}

//
void zmqClient::zmqGet(float* weight, int num){
	assert(weight != NULL);
	zmqprotobuf::PSMSG msg;
	msg.set_type(PULL_REQUEST); // 1 for pull, 2 for push
	msg.mutable_weights()->Reserve(num);
	for (int i = 0; i < num; ++ i){
		msg.add_weights(0);
	}
	zmqGet(msg);
	assert(num == msg.weights_size());
	for (int i = 0; i < num; ++ i){
		weight[i] = msg.weights(i);
	}
}
void zmqClient::zmqPut(zmqprotobuf::PSMSG & pmsg){
	// send gradients to server
	std::string outmsg = pmsg.SerializeAsString();
	zmq::message_t request (outmsg.length());
	memcpy ((void *) request.data (), outmsg.data(), outmsg.length());
	// send the message through socket
	socket->send (request);

	// reply from server to indicate gradients received
	zmq::message_t reply;
	socket->recv (&reply);
}

void zmqClient::zmqPut(float* derv, int num){
	assert(derv != NULL);
	zmqprotobuf::PSMSG msg;
	msg.set_type(PUSH_REQUEST); // 1 for pull, 2 for push
	msg.mutable_weights()->Reserve(num);
	for (int i = 0; i < num; ++ i){
		msg.add_gradients(derv[i]);
	}

	zmqPut(msg);
}
zmqClient::~zmqClient(void)
{
	socket->close();
	context->close();
}
