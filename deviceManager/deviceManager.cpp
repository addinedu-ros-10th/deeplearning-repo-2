#include "deviceManager.hpp"
#include <iostream>

DeviceManager::~DeviceManager()
{
    if (sound) {
        sound->release();
        sound = nullptr;
    }
    if (fmodsystem) {
        fmodsystem->close();
        fmodsystem->release();
        fmodsystem = nullptr;
    }
}

void DeviceManager::alarm_set()
{
    bool is_playing = false;
    while (true) {
        current_alarm = get_alarm();
        last_alarm = get_last_alarm();

        std::cout << "Current Alarm: " << int(current_alarm.load()) << "\n";
        switch (current_alarm.load())
        {
            case 0x00:
                break;
            case 0xFF:
                std::cout << "Alarm Disable \n";
                // stop_sound();
                break;
            case 0x01:
                std::cout << "Fire Alarm \n";
                set_sound("fire.mp3");
                break;
            case 0x02:
                std::cout << "Violence Alarm \n";
                set_sound("violence.mp3");
                break;
            case 0x03:
                std::cout << "Trash Alarm \n";
                set_sound("trash.mp3");
                break;
            case 0x04:
                std::cout << "Smoking Alarm \n";
                set_sound("smoke.mp3");
                break;
            default:
                std::cout << "Unknown Alarm \n";
                break;
        }

        channel->isPlaying(&is_playing);
        if (channel->isPlaying(&is_playing) != FMOD_OK) {
            std::cout << "System Check Failed" << std::endl;
        }

        if (is_playing) {
            std::cout << "Sound Playing" << std::endl;
        }
        else
        {
            if (last_alarm.load() == 0 || last_alarm.load() == 0xFF){}
            else if (last_alarm.load() == 1 || last_alarm.load() == 2)
            {
                play_sound();
                std::cout << "Playing Sound \n";
            }
            else
            {
                play_sound();
                std::this_thread::sleep_for(std::chrono::seconds(30));
                save_data(0xFF);
            }
        }
        fmodsystem->update();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

uint8_t DeviceManager::get_alarm()
{
    return load_data();
}

uint8_t DeviceManager::get_last_alarm()
{
    current_alarm.store(get_alarm());
    if (current_alarm != 0) {
        last_alarm.store(current_alarm);
    }
    return last_alarm.load();
}

void DeviceManager::set_sound(const char* sound_file)
{
    std::string full_path = base_path + "/" + sound_file;
    if (sound) {
        sound->release();
        sound = nullptr;
    }
    if (fmodsystem->createSound(full_path.c_str(), FMOD_DEFAULT, 0, &sound) != FMOD_OK) {
        std::cerr << "Failed to load sound: " << full_path << std::endl;
    }
}

void DeviceManager::play_sound()
{
    if (sound) {
        if (fmodsystem->playSound(sound, 0, false, &channel) != FMOD_OK) {
            std::cerr << "Failed to play sound." << std::endl;
        }
    } else {
        std::cerr << "No sound loaded to play." << std::endl;
    }
}

void DeviceManager::stop_sound()
{
    if (channel) {
        bool is_playing = false;
        if (channel->isPlaying(&is_playing) == FMOD_OK && is_playing) {
            channel->stop();
        }
    }
}

void DeviceManager::start()
{
    socket_init();
    std::thread receive_thread(&TCPClientManager::receive_data, this);
    std::thread alarm_thread(&DeviceManager::alarm_set, this);

    receive_thread.join();
    alarm_thread.join();
}

int main()
{
    DeviceManager deviceManager("127.0.0.1", 1201, 0x01, 0x01, 0x02);
    deviceManager.start();
    return 0;
}