#ifndef OPENCL_UTILS_HPP
#define OPENCL_UTILS_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <string>
#include <CL/cl.hpp>

inline cl::Platform get_platform()
{

    /* Get available platforms.
     * This is an equivalent of clinfo in shell */
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cerr << "No platform found." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Using the first available platform
    cl::Platform platform = platforms.at(0);
    // Display platform name using C++ bindings wrapping clGetPlatformInfo function
    std::cerr << "Using platform " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    return platform;
}

inline cl::Device get_device(cl::Platform platform)
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
    std::cerr << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    return device;
}

inline cl::Program load_and_build_program(cl::Context context, cl::Device device, std::string kernel_source_file)
{
    std::ifstream kernel_source(kernel_source_file);
    if (kernel_source.is_open())
    {
        std::string source_code;
        /* For efficiency purposes, preallocate the string by going to the end of file
         * using seekg, getting the size using tellg and then go back to the beginning */
        kernel_source.seekg(0, std::ios::end);   
        source_code.reserve(kernel_source.tellg());
        kernel_source.seekg(0, std::ios::beg);
        /* Read source file into string using streambuf_iterator 
         * Extra parenthesis to constructor due to the "most vexing parse" */
        source_code.assign((std::istreambuf_iterator<char>(kernel_source)), 
                            std::istreambuf_iterator<char>());
        kernel_source.close();
        /* Create the program using the source code and the context */
        cl::Program program(context, cl::Program::Sources(1, std::make_pair(source_code.c_str(), source_code.length())));
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
        std::cerr << "Could not load kernel source code located in " << kernel_source_file << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif