#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include <algorithm>

#define MAX_WIDTH 4096 
#define KERNEL_SIZE 3
#define WIDE_BUS_WIDTH 512 
#define PIXELS_PER_BURST (WIDE_BUS_WIDTH / 32) 

typedef ap_uint<8> PIXEL_TYPE;
typedef ap_uint<32> BUS_TYPE; 
typedef ap_uint<WIDE_BUS_WIDTH> WIDE_BUS_TYPE;

PIXEL_TYPE grayscale_weighted(PIXEL_TYPE r, PIXEL_TYPE g, PIXEL_TYPE b) {
    #pragma HLS INLINE
    return (r * 77 + g * 150 + b * 29) >> 8; 
}

void read_and_grayscale(
    const WIDE_BUS_TYPE* in_img, 
    hls::stream<PIXEL_TYPE>& stream_grayscale, 
    int total_bursts) 
{
    READ_BURST_LOOP: 
    for (int i = 0; i < total_bursts; i++) {
        #pragma HLS PIPELINE II=1
        WIDE_BUS_TYPE wide_data = in_img[i];

        UNPACK_PIXELS:
        for (int p = 0; p < PIXELS_PER_BURST; p++) {
            #pragma HLS UNROLL
            BUS_TYPE pixel_32 = wide_data((p + 1) * 32 - 1, p * 32);
            PIXEL_TYPE r = (pixel_32 >> 16) & 0xFF;
            PIXEL_TYPE g = (pixel_32 >> 8) & 0xFF;
            PIXEL_TYPE b = pixel_32 & 0xFF;
            
            stream_grayscale.write(grayscale_weighted(r, g, b));
        }
    }
}


void sobel_process(
    hls::stream<PIXEL_TYPE>& stream_grayscale, 
    hls::stream<PIXEL_TYPE>& stream_edge_output, 
    int height, 
    int width) 
{
    PIXEL_TYPE line_buffer[KERNEL_SIZE - 1][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1 
    #pragma HLS DEPENDENCE variable=line_buffer array inter false 
    
    PIXEL_TYPE window[KERNEL_SIZE][KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0 

    const int S_KX[KERNEL_SIZE][KERNEL_SIZE] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    const int S_KY[KERNEL_SIZE][KERNEL_SIZE] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };
    const int MAG_SCALE_SHIFT = 1; 
    
    int total_pixels = height * width;

    CLEAR_BUFFERS_LOOP_OUTER:
    for (int k = 0; k < KERNEL_SIZE - 1; k++) {
        #pragma HLS UNROLL
        CLEAR_BUFFERS_LOOP_INNER:
        for (int l = 0; l < MAX_WIDTH; l++) {
            #pragma HLS PIPELINE II=1
            line_buffer[k][l] = 0;
        }
    }

    SOBEL_PROCESS_LOOP:
    for (int i = 0; i < total_pixels; i++) {
        #pragma HLS PIPELINE II=1
        
        PIXEL_TYPE gray_pixel = stream_grayscale.read(); 

        int row = i / width;
        int col = i % width;

        WINDOW_SHIFT_OUTER:
        for (int k = 0; k < KERNEL_SIZE; k++) {
            #pragma HLS UNROLL 
            WINDOW_SHIFT_INNER:
            for (int l = 0; l < KERNEL_SIZE - 1; l++) {
                #pragma HLS UNROLL 
                window[k][l] = window[k][l + 1];
            }
        }
        for (int k = 0; k < KERNEL_SIZE - 1; k++) {
            #pragma HLS UNROLL
            window[k][KERNEL_SIZE - 1] = line_buffer[k][col];
            line_buffer[k][col] = window[k+1][KERNEL_SIZE - 1];
        }
        window[KERNEL_SIZE - 1][KERNEL_SIZE - 1] = gray_pixel;
        
        if (row >= 1 && col >= 1 && row < height - 1 && col < width - 1) {
            
            int Gx = 0;
            int Gy = 0;
            
            CONVOLUTION_OUTER:
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                #pragma HLS UNROLL 
                CONVOLUTION_INNER:
                for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                    #pragma HLS UNROLL 
                    Gx += window[kr][kc] * S_KX[kr][kc]; 
                    Gy += window[kr][kc] * S_KY[kr][kc];
                }
            }

            int magnitude = abs(Gx) + abs(Gy); 
            int scaled_magnitude = magnitude >> MAG_SCALE_SHIFT;

            PIXEL_TYPE edge_pixel = (scaled_magnitude > 255) ? 255 : (scaled_magnitude < 0) ? 0 : scaled_magnitude;
            stream_edge_output.write(edge_pixel);
        }
    }
}

void write_and_pack(
    WIDE_BUS_TYPE* out_img, 
    hls::stream<PIXEL_TYPE>& stream_edge_output, 
    int output_pixels_to_write) 
{
    int output_bursts = (output_pixels_to_write + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST;

    WRITE_BURST_LOOP:
    for (int i = 0; i < output_bursts; i++) {
        #pragma HLS PIPELINE II=1
        WIDE_BUS_TYPE wide_data = 0;
        int current_pixel_index = i * PIXELS_PER_BURST;

        PACK_PIXELS:
        for (int p = 0; p < PIXELS_PER_BURST; p++) {
            #pragma HLS UNROLL
            if (current_pixel_index + p < output_pixels_to_write) {
                PIXEL_TYPE edge_pixel = stream_edge_output.read(); 

                BUS_TYPE pixel_32 = (edge_pixel << 16) | (edge_pixel << 8) | edge_pixel;
                wide_data((p + 1) * 32 - 1, p * 32) = pixel_32;
            }
        }
        out_img[i] = wide_data;
    }
}


extern "C" {
void image_process(
    const WIDE_BUS_TYPE* in_img,
    WIDE_BUS_TYPE* out_img,
    int height,
    int width)
{
#pragma HLS INTERFACE m_axi port=in_img   offset=slave bundle=gmem0 
#pragma HLS INTERFACE m_axi port=out_img  offset=slave bundle=gmem1 
#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=return

    #pragma HLS DATAFLOW 

    hls::stream<PIXEL_TYPE> stream_grayscale("grayscale_stream");
    hls::stream<PIXEL_TYPE> stream_edge_output("edge_output_stream");

    int total_pixels = height * width;
    int total_bursts = (total_pixels + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST;
    int output_pixels_to_write = (width - 2) * (height - 2); 

    read_and_grayscale(in_img, stream_grayscale, total_bursts);
    sobel_process(stream_grayscale, stream_edge_output, height, width);
    write_and_pack(out_img, stream_edge_output, output_pixels_to_write);
}
}
