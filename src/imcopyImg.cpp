#include <opencv2/opencv.hpp>

#include "opencl_utils.hpp"
#include "benchmark.hpp"

int main (int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " src_file.*" << "dst_file.*" << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Load image file */
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    size_t height = image.rows;
    size_t width = image.cols;

    /* Convert it to RGBA */
    cv::Mat image_rgba;
    cv::cvtColor(image, image_rgba, CV_BGRA2RGBA);

    // Get pixel data array
    int size = strlen(reinterpret_cast<char*>(image_rgba.data));
    uchar* pixels = new uchar[width * height * 4];
    pixels = image_rgba.data;

    cl::Platform platform = get_platform();
    cl::Device device = get_device(platform);

    cl::Context runtimeContext({device});
    cl::Program program = load_and_build_program(runtimeContext, device, "../src/kernelCopy.cl");

    // SEGFAULT: 
    cl::ImageFormat format(CL_RGBA, CL_UNSIGNED_INT8);
    // cl::Image2D IMAGE(runtimeContext, 
    //                 CL_MEM_READ_ONLY | CL_MEM_HOST_PTR, 
    //                 format, 
    //                 (uint)width, 
    //                 (uint)height, 
    //                 0, 
    //                 (void*) pixels);
    
    // cl::Image2D IMAGE_OUT(runtimeContext,
    //                     CL_MEM_WRITE_ONLY,
    //                     format,
    //                     (uint)width,
    //                     (uint)height);
    
}