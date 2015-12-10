# DistributedLigisticRegression
This is a distributed implementation of logistic regression algorithm.

This includes two parts: 

# client

The client compute gradients based on its subet of data using weights pull from server, and send new gradients to server

# server

The server takes gradients from clients, update weights, and send weights to clients if requested by clients.

# Implementation:

The communication is managed by ZeroMQ (http://zeromq.org/), and message serialization/deserialization is done by 
Goolgle Protobuf (https://github.com/google/protobuf).

The server side code looks like this:

'''c

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
'''

# Compile

The code is compiled in Windows Visual Studio. Windows version of Goolgle protobuf and Zeromq are required. 
