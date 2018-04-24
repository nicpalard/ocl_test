#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <string>
#include <cassert>
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>

#include "benchmark.hpp"

cl::Platform pickUpPlatform()
{
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    /* Get available platforms.
     * This is an equivalent of clinfo in shell */
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cerr << "No platform found." << std::endl;
        exit(EXIT_FAILURE);  
    }
    
    /* Using the first available platform */
    cl::Platform platform = platforms.at(0);
    /* Display platform name using C++ bindings wrapping clGetPlatformInfo function */
    std::cerr << "Using platform " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    return platform;
}

cl::Device pickUpDevice(cl::Platform platform)
{
    /* Get available devices from platform. */
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0)
    {
        std::cerr << "No device found." << std::endl;
    }

    /* Using the first available device */
    cl::Device device = devices.at(0);
    std::cout<< "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    return device;
}

cl::Program loadAndBuildProgram(cl::Context context, cl::Device device, std::string fileName)
{
    std::ifstream kernelSource(fileName);
    if (kernelSource.is_open())
    {
        std::string sourceCode;
        /* For efficiency purposes, preallocate the string by going to the end of file
         * using seekg, getting the size using tellg and then go back to the beginning */
        kernelSource.seekg(0, std::ios::end);   
        sourceCode.reserve(kernelSource.tellg());
        kernelSource.seekg(0, std::ios::beg);
        /* Read source file into string using streambuf_iterator 
         * Extra parenthesis to constructor due to the "most vexing parse" */
        sourceCode.assign((std::istreambuf_iterator<char>(kernelSource)), 
                                std::istreambuf_iterator<char>());
        kernelSource.close();
        /* Create the program using the source code and the context */
        cl::Program program(context, cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length())));
        try
        {
            program.build({device});   
        }
        catch(cl::Error e)
        {
            std::cerr << "Could not build program: "<< std::endl
                        << "\tDevice name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl
                        << "\tStatus code: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)  << std::endl
                        << "Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            exit(EXIT_FAILURE);
        }
        return program;    
    }
    else
    {
        std::cerr << "Could not load kernel source code located in " + fileName << std::endl;
        exit(EXIT_FAILURE);
    }

}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " image_file.*" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    /* Load image file */
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    size_t height = image.rows;
    size_t width = image.cols;

    /* Convert it to RGBA */
    cv::Mat image_rgba;
    cv::cvtColor(image, image_rgba, CV_BGR2RGBA);
    
    // Get pixel data array
    int size = strlen(reinterpret_cast<char*>(image_rgba.data));
    uchar* pixels = new uchar[width * height * 4];
    pixels = image_rgba.data;

    /* Debug informations */
    std::cerr << "Loaded " << argv[1] << " - " << width << "x" << height << " - " << size * sizeof(char) / 1e6 << " MB" << std::endl;
    std::cerr << "Old size: " << strlen((char*)(reinterpret_cast<char*>(image.data))) << " - Expected size: " << width * height * 4 << " - Real size: " << size << std::endl;

    cl::Platform platform = pickUpPlatform();
    cl::Device device = pickUpDevice(platform);

    cl::Context runtimeContext({device});
    cl::Program program = loadAndBuildProgram(runtimeContext, device, "../src/kernelCopy.cl");

    // Create input and output image buffer
    // TODO: Use cl::Image2D (not working for now)
    cl::Buffer IMAGE(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4 * sizeof(uchar), pixels);
    cl::Buffer OUT_IMAGE(runtimeContext, CL_MEM_WRITE_ONLY, width * height * 4 * sizeof(uchar));
    
    cl::Kernel copyKernel(program, "copy");
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
    cv::cvtColor(cv::Mat(height, width, CV_8UC4, copy_pixels), result, CV_RGBA2BGR);
    cv::imwrite("test01.png", result);

    image.release();
    result.release();
}