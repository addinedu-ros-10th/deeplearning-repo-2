#pragma once

#ifndef TCPCLIENT_HPP
#define TCPCLIENT_HPP
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>
#include <string>
#include <atomic>
#include <unistd.h>

using namespace std;

typedef struct comm_interface{
    uint8_t source;
    uint8_t destination;
    uint8_t patrol_number;
    uint8_t alarm_type;
} comm_interface;

class TCPClientManager {
    private:
        std::atomic<comm_interface> recv_data;
        uint8_t PATROL_NUMBER;
        uint8_t deviceManager_ID;
        uint8_t situationDetector_ID;
        sockaddr_in server_addr;
        int client_socket;

public:
    TCPClientManager(char* host, int port, uint8_t patrol_number, uint8_t device_id, uint8_t detector_id);
    ~TCPClientManager();

    void socket_init();
    void socket_close();
    
    void send_data(uint8_t *data, size_t length);

    void receive_data();
    void date_validation();

    uint8_t load_data();
    void save_data(uint8_t data);
};

#endif // DEVICEMANAGER_HPP