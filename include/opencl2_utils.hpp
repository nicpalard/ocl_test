#ifndef OPENCL2_UTILS_HPP
#define OPENCL2_UTILS_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS

#include <iostream>
#include <CL/cl2.hpp>

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

    cl::Platform platform;
    for (auto &p : platforms) {
        std::string platform_version = p.getInfo<CL_PLATFORM_VERSION>();
        if (platform_version.find("OpenCL 2.") != std::string::npos) {
            platform = p;
            break;
        }
    }

    if (platform() == 0)  {
        std::cerr << "No OpenCL 2.0 platform found." << std::endl;
        throw std::runtime_error("No OpenCL 2.0 platform found.");
    }
    return platform;
}

inline void set_default_platform(cl::Platform plat)
{
    if (cl::Platform::setDefault(plat) != plat)
    {
        std::cerr << "Error while setting default platform." << std::endl;
        throw std::runtime_error("Unable to set default platform.");
    }
}

#endif