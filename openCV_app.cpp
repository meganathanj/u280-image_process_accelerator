#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono> 
#include <iomanip> 
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;
using namespace std::chrono;

#define IMG_HEIGHT 1080 
#define IMG_WIDTH 1920  
#define TOTAL_PIXELS (IMG_HEIGHT * IMG_WIDTH)

///@brief: Function that performs edge detection on input image
void process_image_opencv(const cv::Mat& input_img, cv::Mat& output_img) {
    
    cv::Mat gray_img;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    cv::cvtColor(input_img, gray_img, cv::COLOR_BGR2GRAY);

    cv::Sobel(gray_img, grad_x, CV_8U, 1, 0, 3);
    cv::Sobel(gray_img, grad_y, CV_8U, 0, 1, 3);
    
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    const double alpha_scale = 1.0; 
    const double beta_scale = 1.0; 
    const double gamma_value = 0.0; 

    cv::addWeighted(abs_grad_x, alpha_scale, abs_grad_y, beta_scale, gamma_value, output_img);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_dir> <output_dir>" << std::endl;
        std::cout << "Note: This is the CPU/OpenCV version, no XCLBIN needed." << std::endl;
        return EXIT_FAILURE;
    }

    fs::path input_dir_path = argv[1];
    fs::path output_dir_path = argv[2];

    if (!fs::exists(output_dir_path)) {
        fs::create_directories(output_dir_path);
    }

    std::vector<fs::path> image_files;
    for (const auto& entry : fs::directory_iterator(input_dir_path)) {
        if (entry.is_regular_file()) {
            image_files.push_back(entry.path());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    int num_images = image_files.size();
    if (num_images == 0) {
        std::cerr << "[ERROR] No input images found in: " << input_dir_path << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "[INFO] Starting OpenCV batch process for " << num_images << " images..." << std::endl;

    /// @breif: Determine performance metrics and print on the console
    double total_process_time_ms = 0.0;
    double total_io_time_ms = 0.0; 
    auto t_start_full = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        fs::path current_input_path = image_files[i];
        fs::path output_path = output_dir_path / ("out_cpu_" + current_input_path.filename().string());

        auto t_io_start = high_resolution_clock::now();

        cv::Mat input_img = cv::imread(current_input_path.string(), cv::IMREAD_COLOR);
        
        if (input_img.empty()) {
            std::cerr << "[ERROR] Could not read image: " << current_input_path.string() << std::endl;
            continue;
        }

        auto t_proc_start = high_resolution_clock::now();
        cv::Mat output_img;
        process_image_opencv(input_img, output_img);
        auto t_proc_end = high_resolution_clock::now();
        
        cv::imwrite(output_path.string(), output_img);
        auto t_io_end = high_resolution_clock::now();

        total_process_time_ms += duration<double, std::milli>(t_proc_end - t_proc_start).count();
        total_io_time_ms += duration<double, std::milli>(t_io_end - t_io_start).count();
    }

    auto t_end_full = high_resolution_clock::now();
    duration<double, std::milli> total_full_time_ms = t_end_full - t_start_full;

    std::cout << "\n=================================================" << std::endl;
    std::cout << "[SUCCESS] OPENCV PREPROCESSING COMPLETED" << std::endl;
    std::cout << "[SUMMARY] TOTAL IMAGES PROCESSED: " << num_images <<std::endl;
    
    double total_pixels_processed = (double)num_images * TOTAL_PIXELS;
    double avg_process_time_s = (total_process_time_ms / num_images) / 1000.0;
    double avg_io_time_s = (total_io_time_ms / num_images) / 1000.0;
    
    double process_throughput_mpps = (TOTAL_PIXELS / avg_process_time_s) / 1000000.0;
    double io_throughput_mpps = (TOTAL_PIXELS / avg_process_time_s) / 1000000.0; 
    
    std::cout << "\n===== PERFORMANCE SUMMARY (" << num_images << " Images) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << std::left << std::setw(35) << "TOTAL RUNTIME (I/O + PROCESSING)" << total_full_time_ms.count() << " ms" << std::endl;
    std::cout << std::left << std::setw(35) << "CPU Processing Time:" << total_process_time_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(35) << "I/O Time:" << total_io_time_ms << " ms" << std::endl;
    
    std::cout << "\n--- PER IMAGE THROUGHPUT ---" << std::endl;
    std::cout << std::left << std::setw(35) << "CPU THROUGHPUT: " << process_throughput_mpps << " MPPS" << std::endl;
    std::cout << std::left << std::setw(35) << "I/O THROUGHPUT: " << io_throughput_mpps << " MPPS" << std::endl;

    return EXIT_SUCCESS;
}
