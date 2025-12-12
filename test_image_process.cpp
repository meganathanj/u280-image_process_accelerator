#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <ap_int.h>

#define MAX_WIDTH 4096 
#define WIDE_BUS_WIDTH 512
#define PIXELS_PER_BURST (WIDE_BUS_WIDTH / 32) 

typedef ap_uint<8> PIXEL_TYPE;
typedef ap_uint<32> BUS_TYPE; 
typedef ap_uint<WIDE_BUS_WIDTH> WIDE_BUS_TYPE;

extern "C" {
void image_process(
    const WIDE_BUS_TYPE* in_img, 
    WIDE_BUS_TYPE* out_img,      
    int height,
    int width);
}

void pack_image_data(const BUS_TYPE* unpacked, WIDE_BUS_TYPE* packed, int size) {
    int total_bursts = (size + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST;

    for (int i = 0; i < total_bursts; i++) {
        WIDE_BUS_TYPE wide_data = 0;
        for (int p = 0; p < PIXELS_PER_BURST; p++) {
            int pixel_idx = i * PIXELS_PER_BURST + p;
            if (pixel_idx < size) {
                wide_data((p + 1) * 32 - 1, p * 32) = unpacked[pixel_idx];
            }
        }
        packed[i] = wide_data;
    }
}

void unpack_image_data(const WIDE_BUS_TYPE* packed, BUS_TYPE* unpacked, int size) {
    int total_bursts = (size + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST;
    
    for (int i = 0; i < total_bursts; i++) {
        WIDE_BUS_TYPE wide_data = packed[i];
        for (int p = 0; p < PIXELS_PER_BURST; p++) {
            int pixel_idx = i * PIXELS_PER_BURST + p;
            if (pixel_idx < size) {
                unpacked[pixel_idx] = wide_data((p + 1) * 32 - 1, p * 32);
            }
        }
    }
}


int main() {
    const int HEIGHT = 64;
    const int WIDTH = 64;
    const int INPUT_SIZE = HEIGHT * WIDTH;
    const int OUTPUT_SIZE = (HEIGHT - 2) * (WIDTH - 2); 

    std::cout << "--- Starting HLS Test Bench for Wide AXI Kernel V3 ---" << std::endl;
    std::cout << "Image size: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Output size: " << (WIDTH - 2) << "x" << (HEIGHT - 2) << std::endl;

    BUS_TYPE *input_32bit = new BUS_TYPE[INPUT_SIZE];
    BUS_TYPE *output_32bit = new BUS_TYPE[OUTPUT_SIZE];
    
    std::cout << "Generating synthetic input data..." << std::endl;
    for (int i = 0; i < INPUT_SIZE; i++) {
        unsigned char val = (unsigned char)(i % 256);
        input_32bit[i] = (val << 16) | (val << 8) | val;
    }

    const int INPUT_BURSTS = (INPUT_SIZE + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST;
    const int OUTPUT_BURSTS = (OUTPUT_SIZE + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST;
    
    WIDE_BUS_TYPE *input_wide = new WIDE_BUS_TYPE[INPUT_BURSTS];
    WIDE_BUS_TYPE *output_wide = new WIDE_BUS_TYPE[OUTPUT_BURSTS];

    pack_image_data(input_32bit, input_wide, INPUT_SIZE);

    std::cout << "Calling image_process kernel..." << std::endl;
    image_process(input_wide, output_wide, HEIGHT, WIDTH);
    std::cout << "Kernel execution complete." << std::endl;

    unpack_image_data(output_wide, output_32bit, OUTPUT_SIZE);

    int errors = 0;
    if (OUTPUT_SIZE > 0) {
        if (output_32bit[0] == 0) {
            std::cerr << "Verification FAILED: First output pixel is 0. Sobel output expected to be non-zero for a gradient input." << std::endl;
            errors++;
        }
    }

    try {
        delete[] input_32bit;
        delete[] output_32bit;
        delete[] input_wide;
        delete[] output_wide;
    } catch (...) {
        std::cerr << "CRITICAL ERROR: Memory cleanup failed." << std::endl;
        errors++;
    }

    if (errors == 0) {
        std::cout << "--- HLS C Simulation PASSED (Wide AXI Kernel) ---" << std::endl;
        return 0;
    } else {
        std::cout << "--- HLS C Simulation FAILED ---" << std::endl;
        return 1;
    }
}
