#pragma once

#ifndef DEVICEMANAGER_HPP
#define DEVICEMANAGER_HPP
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>
#include <string>
#include <atomic>

using namespace std;

typedef struct device_info{

} device_info;

typedef struct comm_interface{
    u_int8_t source;
    u_int8_t destination;
    u_int8_t patrol_number;
    u_int8_t alarm_type;
} comm_interface;

class TCPClientManager {
    private:
        u_int8_t PATROL_NUMBER;
        u_int8_t deviceManager_ID;
        sockaddr_in server_addr;
        int client_socket;

public:
    TCPClientManager(char* host, int port, u_int8_t patrol_number, u_int8_t id);
    void socket_init();
    void receive_data();
    void send_data();
};
 
TCPClientManager::TCPClientManager(char* host, int port, u_int8_t patrol_number, u_int8_t id)
{
    PATROL_NUMBER = patrol_number;
    deviceManager_ID = id;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(host);
}

void TCPClientManager::socket_init()
{
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if(client_socket < 0){
        cerr << "Socket creation error" << endl;
        exit(EXIT_FAILURE);
    }

    if(connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0){
        cerr << "Connection Failed" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Connected to server" << endl;
}

void TCPClientManager::receive_data()
{
    char buffer[1024] = {0};
    while(true){
        int valread = read(client_socket, buffer, 1024);
        if(valread > 0){
            cout << "Message from server: " << buffer << endl;
        }
    }
}



#endif // DEVICEMANAGER_HPP