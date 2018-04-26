#include <opencv2/opencv.hpp>

#include "opencl_utils.hpp"
#include "benchmark.hpp"

int main(int argc, char** argv)
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

    /* Debug informations */
    std::cerr << "Loaded " << argv[1] << " - " << width << "x" << height << " - " << size * sizeof(char) / 1e6 << " MB" << std::endl;
    std::cerr << "Old size: " << strlen((char*)(reinterpret_cast<char*>(image.data))) << " - Expected size: " << width * height * 4 << " - Real size: " << size << std::endl;

    cl::Platform platform = get_platform();
    cl::Device device = get_device(platform);

    cl::Context runtimeContext({device});
    cl::Program program = load_and_build_program(runtimeContext, device, "../src/kernelCopy.cl");

    // Create input and output image buffer
    cl::Buffer IMAGE(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4 * sizeof(uchar), pixels);
    cl::Buffer OUT_IMAGE(runtimeContext, CL_MEM_WRITE_ONLY, width * height * 4 * sizeof(uchar));
    
    cl::Kernel copyKernel(program, "copy_buff");
    copyKernel.setArg(0, IMAGE);
    // Kernel scalar arguments can not be of type size_t
    copyKernel.setArg(1, (uint) width);
    copyKernel.setArg(2, OUT_IMAGE);

    /* Creating the command queue that will be used to process */
    cl::CommandQueue queue(runtimeContext, device);
    /* Launch the kernel on the compute device */
    queue.enqueueNDRangeKernel(copyKernel, cl::NullRange, cl::NDRange(width, height, 4), cl::NullRange);

    uchar* copy_pixels = new uchar[width * height * 4];
    /* Get the result back to host */
    queue.enqueueReadBuffer(OUT_IMAGE, CL_TRUE, 0, width * height * 4 * sizeof(uchar), copy_pixels);

    cv::Mat result;
    cv::cvtColor(cv::Mat(height, width, CV_8UC4, copy_pixels), result, CV_RGBA2BGRA);
    cv::imwrite(argv[2], result);

    image.release();
    result.release();
}



























