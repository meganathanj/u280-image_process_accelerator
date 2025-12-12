///@brief: C++ Headers for core logic
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring> 
#include <CL/cl_ext_xilinx.h> 
#include <chrono> 
#include <filesystem> 
#include <iomanip>    
#include <cmath>    

///@brief: XRT ultitiy includes
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

///@brief: openCV ultitiy includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
typedef uint32_t Pixel;

///@brief: Input images MAX bounds
#define IMG_HEIGHT 1080 
#define IMG_WIDTH 1920  
#define TOTAL_PIXELS (IMG_HEIGHT * IMG_WIDTH)

void save_output_image(const std::vector<unsigned int>& buffer, int out_height, int out_width, const std::string& output_path) {
    
    cv::Mat output_image(out_height, out_width, CV_8UC1); 
    int out_size = out_height * out_width;

    for (int i = 0; i < out_size; i++) {
        unsigned char edge_pixel = buffer[i] & 0xFF; 
        int r = i / out_width; 
        int c = i % out_width;
        output_image.at<unsigned char>(r, c) = edge_pixel;
    }
    cv::imwrite(output_path, output_image);
}

struct PerformanceMetrics {
    double h2d_time_ms = 0.0;
    double kernel_time_ms = 0.0;
    double d2h_time_ms = 0.0;
    double total_time_ms = 0.0;
    int pixels_processed = 0; 
};

///@breif: Function to handle image pixel transfer
PerformanceMetrics process_image_fpga(xrt::kernel& kernel, xrt::device& device, const std::string& input_path, const std::string& output_path) {
    
    PerformanceMetrics metrics;
    cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[ERROR] COULD NOT LOAD IMAGE: " << input_path << std::endl; // LOG CAPITALIZED
        return metrics;
    }

    int height = image.rows;
    int width = image.cols;
    int size = height * width; 
    
    /// Keep track of the pixels that were transfered in
    metrics.pixels_processed = size; 

    int out_height = height - 2;
    int out_width = width - 2;
    int out_size = out_height * out_width;

    /// Prepare a vector of pixels
    std::vector<unsigned int> input_vector(size);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j); 
            input_vector[i * width + j] = (pixel[2] << 16) | (pixel[1] << 8) | pixel[0]; 
        }
    }
    
    /// XRT initialization
    size_t bo_size_bytes = size * sizeof(unsigned int); 
    
    xrt::bo bo_in  = xrt::bo(device, bo_size_bytes, xrt::bo::flags::cacheable, kernel.group_id(0));
    xrt::bo bo_out = xrt::bo(device, bo_size_bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
    
    unsigned int* bo_in_map  = bo_in.map<unsigned int*>();
    
    /// Handle transfering pixel vectors and obtaining processed vectors back
    auto h2d_start = std::chrono::high_resolution_clock::now(); 
    std::copy(input_vector.begin(), input_vector.end(), bo_in_map);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto h2d_stop = std::chrono::high_resolution_clock::now();
    metrics.h2d_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(h2d_stop - h2d_start).count() / 1000.0;

    auto kernel_start = std::chrono::high_resolution_clock::now(); 
    auto run = kernel(bo_in, bo_out, height, width);
    run.wait(); 
    auto kernel_stop = std::chrono::high_resolution_clock::now(); 
    metrics.kernel_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(kernel_stop - kernel_start).count() / 1000.0;
    
    auto d2h_start = std::chrono::high_resolution_clock::now(); 
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE); 
    auto d2h_stop = std::chrono::high_resolution_clock::now(); 
    metrics.d2h_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(d2h_stop - d2h_start).count() / 1000.0;

    metrics.total_time_ms = metrics.h2d_time_ms + metrics.kernel_time_ms + metrics.d2h_time_ms;
    
    unsigned int* bo_out_map = bo_out.map<unsigned int*>(); 
    std::vector<unsigned int> output_vector(out_size); 

    std::copy(bo_out_map, bo_out_map + out_size, output_vector.data());
    
    save_output_image(output_vector, out_height, out_width, output_path);    
    return metrics;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "USAGE: " << argv[0] << " <XCLBIN_PATH> <INPUT_DIR> <OUTPUT_DIR>" << std::endl;
        return 1;
    }
    
    const std::string xclbin_path = argv[1];
    const std::string input_dir = argv[2];
    const std::string output_dir = argv[3];
    fs::create_directories(output_dir);
    
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "[INFO] INITIALIZING XRT DEVICE AND LOADING XCLBIN..." << std::endl; 
    auto setup_start = std::chrono::high_resolution_clock::now();
    try {
        auto device = xrt::device(0); 
        auto uuid = device.load_xclbin(xclbin_path);
        auto kernel = xrt::kernel(device, uuid, "image_process"); 
        auto setup_stop = std::chrono::high_resolution_clock::now();
        double setup_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(setup_stop - setup_start).count() / 1000.0;
        
        std::cout << "[INFO] XRT SETUP/LOAD TIME: " << setup_time_ms << " MS" << std::endl; 
        std::cout << "=================================================" << std::endl;

        double total_h2d_time_ms = 0.0;
        double total_kernel_time_ms = 0.0;
        double total_d2h_time_ms = 0.0;
        double total_end_to_end_time_ms = 0.0;
        long long total_input_pixels = 0; 
        int image_count = 0;

        std::cout << "[INFO] STARTING BATCH PROCESSING FROM: " << input_dir << std::endl; 

        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file() && 
                (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" || entry.path().extension() == ".png")) 
            {
                std::string input_file = entry.path().string();
                std::string output_file = output_dir + "/out_fpga_" + entry.path().filename().string();
                
                PerformanceMetrics metrics = process_image_fpga(kernel, device, input_file, output_file);
                
                if (metrics.kernel_time_ms > 0.0) { 
                    total_h2d_time_ms += metrics.h2d_time_ms;
                    total_kernel_time_ms += metrics.kernel_time_ms;
                    total_d2h_time_ms += metrics.d2h_time_ms;
                    total_end_to_end_time_ms += metrics.total_time_ms;
                    total_input_pixels += metrics.pixels_processed;
                    image_count++;
                }
            }
        }

        /// Handle metrics determination and print it out on console
        if (image_count > 0) {
            
            double avg_h2d_time_s = (total_h2d_time_ms / image_count) / 1000.0;
            double avg_kernel_time_s = (total_kernel_time_ms / image_count) / 1000.0;
            double avg_d2h_time_s = (total_d2h_time_ms / image_count) / 1000.0;
            double avg_total_time_s = (total_end_to_end_time_ms / image_count) / 1000.0;
            
            double avg_pixels_per_image = (double)total_input_pixels / image_count;
            
            double kernel_throughput_mpps = (avg_pixels_per_image / avg_kernel_time_s) / 1000000.0;
            double end_to_end_throughput_mpps = (avg_pixels_per_image / avg_total_time_s) / 1000000.0;
            
            std::cout << "=================================================" << std::endl;
            std::cout << "          FPGA BATCH PERFORMANCE SUMMARY" << std::endl;
            std::cout << "=================================================" << std::endl;
            std::cout << "IMAGES PROCESSED: " << image_count << std::endl;
            std::cout << "--- TOTAL TIMES ---" << std::endl;
            std::cout << std::left << std::setw(25) << "TOTAL H2D DMA TIME:" << total_h2d_time_ms << " MS" << std::endl;
            std::cout << std::left << std::setw(25) << "TOTAL KERNEL TIME:" << total_kernel_time_ms << " MS" << std::endl;
            std::cout << std::left << std::setw(25) << "TOTAL D2H DMA TIME:" << total_d2h_time_ms << " MS" << std::endl;
            std::cout << std::left << std::setw(25) << "TOTAL END-TO-END TIME:" << total_end_to_end_time_ms << " MS" << std::endl;
            std::cout << "--- AVERAGE TIMES PER IMAGE ---" << std::endl;
            std::cout << std::left << std::setw(25) << "AVG H2D DMA TIME:" << total_h2d_time_ms / image_count << " MS" << std::endl;
            std::cout << std::left << std::setw(25) << "AVG KERNEL TIME:" << total_kernel_time_ms / image_count << " MS" << std::endl;
            std::cout << std::left << std::setw(25) << "AVG D2H DMA TIME:" << total_d2h_time_ms / image_count << " MS" << std::endl;
            std::cout << std::left << std::setw(25) << "AVG END-TO-END TIME:" << total_end_to_end_time_ms / image_count << " MS" << std::endl;
            std::cout << "--- THROUGHPUT (MPPS) ---" << std::endl;
            std::cout << std::left << std::setw(25) << "KERNEL THROUGHPUT:" << kernel_throughput_mpps << " MPPS" << std::endl;
            std::cout << std::left << std::setw(25) << "END-TO-END THROUGHPUT:" << end_to_end_throughput_mpps << " MPPS" << std::endl;
            std::cout << "=================================================" << std::endl;
        } else {
            std::cout << "[WARNING] NO IMAGES FOUND IN INPUT DIRECTORY: " << input_dir << std::endl; 
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] XRT/RUNTIME ERROR: " << e.what() << std::endl; 
        return 1;
    }
    return 0;
}
