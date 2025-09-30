#pragma once

#ifndef DEVICEMANAGER_HPP
#define DEVICEMANAGER_HPP
#include "TCPClientManager.hpp"
#include "fmod.hpp"

using namespace std;

class DeviceManager : public TCPClientManager {
    private:
        std::atomic<uint8_t> current_alarm = 0x00;
        std::atomic<uint8_t> last_alarm = 0x00;
        uint8_t PATROL_NUMBER;
        uint8_t deviceManager_ID;
        uint8_t situationDetector_ID;
        sockaddr_in server_addr;
        int client_socket;

        FMOD::System *fmodsystem;
        FMOD::Sound *sound;
        FMOD::Channel *channel = 0;
    
        string base_path;

    public:
        DeviceManager(char* host, int port, uint8_t patrol_number, uint8_t device_id, uint8_t detector_id) : 
            TCPClientManager(host, port, patrol_number, device_id, detector_id),
            fmodsystem(nullptr), sound(nullptr), channel(nullptr), base_path("./embedmediaStorage")
            {
                FMOD::System_Create(&fmodsystem);
                fmodsystem->init(32, FMOD_INIT_NORMAL, 0);
            }
        ~DeviceManager();

        void alarm_set();
        uint8_t get_alarm();
        uint8_t get_last_alarm();
        
        void set_sound(const char* sound_file);
        void play_sound();
        void stop_sound();
        void start();
        void stop();
};

#endif // DEVICEMANAGER_HPP