#include <iostream>
#include <sys/socket.h>
#include <thread>

class TCPClientManager {
    private:
        int TCP_HOST;
        int TCP_PORT;
        u_int8_t PATROL_NUMBER;
        u_int8_t deviceManager_ID;
        
    public:
        void main();
    
}

