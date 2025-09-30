#include "TCPClientManager.hpp"



TCPClientManager::TCPClientManager(char* host, int port, u_int8_t patrol_number, u_int8_t device_id, u_int8_t detector_id)
{
    PATROL_NUMBER = patrol_number;
    deviceManager_ID = device_id;
    situationDetector_ID = detector_id;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(host);
    recv_data.store({0, 0, 0, 0});
}

TCPClientManager::~TCPClientManager()
{
    socket_close();
}


void TCPClientManager::socket_init()
{
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if(client_socket < 0){
        cerr << "Socket creation error \n";
        exit(EXIT_FAILURE);
    }

    while(true){
        if(connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0){
        cout << "situationDetector Connected \n";
        break;
        }
        else{
            cerr << "Connection Failed \n";
            this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
}

void TCPClientManager::socket_close()
{
    if (client_socket >= 0){
        close(client_socket);
        client_socket = -1;
    }
    
}

void TCPClientManager::send_data(uint8_t *data, size_t length)
{
    send(client_socket, data, length, 0);
}

void TCPClientManager::receive_data()
{
    while(true){
        int valread = recv(client_socket, &recv_data, sizeof(recv_data), 0);
        if(valread > 0){
            date_validation();
        }
        
        this_thread::sleep_for(chrono::milliseconds(100));
    }   
}

void TCPClientManager::date_validation()
{
    if (recv_data.load().source == 0x02 && recv_data.load().destination == deviceManager_ID && recv_data.load().patrol_number == PATROL_NUMBER){
        // cout << "Valid Data Received" << endl;
    }
    else{
        cout << "Invalid Data Received" << endl;
        recv_data.store({0, 0, 0, 0});
    }
}

uint8_t TCPClientManager::load_data()
{
    return recv_data.load().alarm_type;
}

void TCPClientManager::save_data(uint8_t data)
{
    recv_data.store({deviceManager_ID, situationDetector_ID, PATROL_NUMBER, data});
}
