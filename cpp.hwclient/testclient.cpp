//
//  Hello World client in C++
//  Connects REQ socket to tcp://localhost:5555
//  Sends "Hello" to server, expects "World" back
//
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

int npush = 2;
int npull = 2;
int nepoches = 1;
int nbatches = 100;

int main ()
{
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);

    std::cout << "Connecting to hello world server..." << std::endl;
    socket.connect ("tcp://localhost:5555");

	long iter = 0;
	for (int epoch = 0; epoch < nepoches; ++ epoch){
		for (int batchid = 0; batchid < nbatches; ++ batchid){
			std::cout << "Iter: " << iter << std::endl;
			if (iter%npull == 0){ // pull from server
				// message contents, complex data structures can be serialized into string format thro
				std::string pull_msg = "pull"; 
				zmq::message_t request (pull_msg.length());
				memcpy ((void *) request.data (), pull_msg.data(), pull_msg.length());
				// send the message through socket
				socket.send (request);
				std::cout << "I just sent a pull request to server, now I am waiting response from server" << std::endl;

				clock_t t_start = clock();
				// waiting from server
				zmq::message_t reply;
				socket.recv (&reply);
				// put the data replied from server into string, then decode it
				std::string reply_msg = std::string(static_cast<char*>(reply.data()), reply.size());
				clock_t t_end = clock();
				std::cout << "It took server "<< (t_end-t_start)/CLOCKS_PER_SEC << " seconds to give me response " << std::endl;

			}

			std::cout << "Now it is my time, I will sleep for a while" << std::endl;
			sleep(1000);

			if (iter%npush == 0){// time to push gradients to server
				std::string push_msg = "push";
				zmq::message_t request (push_msg.length());
				memcpy((void*)request.data(), push_msg.data(), push_msg.length());
				socket.send(request);
				std::cout << "I just sent a push request to server, now I am waiting response from server" << std::endl;
				// just the socket types needs this
				zmq::message_t reply;
				socket.recv (&reply);
				std::string reply_msg = std::string(static_cast<char*>(reply.data()), reply.size());
				std::cout << "Message from server: "<< reply_msg <<  std::endl;
			}

			system("pause");
			++ iter;
		}
	}
	socket.close();
	context.close();
    return 0;
}
