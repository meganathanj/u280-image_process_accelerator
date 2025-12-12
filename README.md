# u280-image_process_accelerator
This project is developed with the goal of accelerating the image preprocessing component involving sobel edge detection on AMD Alveo U280 accelerator tested for the images from BSDS dataset while delivering a higher throughput than a openCV implementation of the same component


PREREQUISTE:

Both the applications {host_app, openCV_app} requires openCV libraries to be configured for successfull execution. Refer to install-openCV.txt to setup and configure dependencies before execution
change permission for the executables using chmod +x host_app if faced with permission conflicts

EXECUTION:
1. Clone the repository
2. unzip test_run cmd: unzip test_run and cd /test_run
3. execute the kernel using the command flow: ./host_app kernelV3.xclbin <INPUT_PATH> <OUTPUT_PATH>
   For example: ./host_app kernelV3.xclbin /test_data /fpga_out

5. Navigate to the corresponding output path specified <OUTPUT_PATH> to obtain the sobel filter processed images


BUILDING THE OPENCV APP:
The openCV app might not execute if the openCV shared objects are not setup correctly! In that case the app can be build with the source code openCV_app.cpp using the below command
   g++ -std=c++17 -Wall     `pkg-config --cflags opencv4`     openCV_app.cpp     -o app     `pkg-config --libs opencv4`
   and the executable can be run using the command: ./openCV_app <INPUT_PATH> <OUTPUT_PATH>
