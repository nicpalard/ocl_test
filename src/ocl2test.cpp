#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "opencl2_utils.hpp"

const int numElements = 32;
int main(void)
{
    cl::Platform platform = get_platform();
    set_default_platform(platform);
}