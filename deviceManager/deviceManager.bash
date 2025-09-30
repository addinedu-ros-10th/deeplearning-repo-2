#! /bin/bash

mkdir build
cd build

cmake ..
make
mv DeviceManager ../deviceManager
cd ..
./deviceManager