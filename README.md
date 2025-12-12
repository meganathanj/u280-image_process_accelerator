# u280-image_process_accelerator
This project is developed with the goal of accelerating the image preprocessing component involving sobel edge detection on AMD Alveo U280 accelerator tested for the images from BSDS dataset while delivering a higher throughput than a openCV implementation of the same component


PREREQUISTE:

Both the applications {host_app, openCV_app} requires openCV libraries to be configured for successfull execution. Refer to install-openCV.txt to setup and configure dependencies before execution
change permission for the executables using chmod +x host_app and chmod +x openCV_app

EXECUTION:
1. Clone the repository
2. unzip test_run using cd /test_run
3. execute the kernel using the command flow: ./host_app kernelV3.xclbin <INPUT_PATH> <OUTPUT_PATH>
   For example: ./host_app kernelV3.xclbin /test_data /fpga_out
4. Execute the openCV application using the command flow: ./openCV_app <INPUT_PATH> <OUTPUT_PATH>
    For example: ./host_app  /test_data /opencv_out

5. Navigate to the corresponding output directories to obtain the processed images
