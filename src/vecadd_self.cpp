#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <streambuf>
#include <CL/cl.hpp>

#include "benchmark.hpp"

void printVector(int* vector, int length)
{
    std::cout << "{" << " ";
    for (int i = 0; i < length; i++)
    {
        std::cout << vector[i] << " ";
    }
    std::cout << "}" << std::endl;
}

void printVector(std::vector<int> vector, int length)
{
    std::cout << "{" << " ";
    for (int i = 0; i < length; i++)
    {
        std::cout << vector.at(i) << " ";
    }
    std::cout << "}" << std::endl;
}

void vectorAdd(std::vector<int> vec1, std::vector<int> vec2, std::vector<int> *out)
{
    for (int i = 0 ; i < vec1.size() ; i++)
    {
        (*out)[i] = vec1[i] + vec2[i];
    }
}

cl::Platform pickUpPlatform()
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
    
    /* Using the first available platform */
    cl::Platform platform = platforms.at(0);
    /* Display platform name using C++ bindings wrapping clGetPlatformInfo function */
    std::cerr << "Using platform " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    return platform;
}


cl::Program loadProgram(cl::Context context, std::string fileName)
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
        return cl::Program(context, cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length())));
    }
    else
    {
        std::cerr << "Could not load kernel source code located in " + fileName << std::endl;
        exit(EXIT_FAILURE);
    }

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

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " vectorSize" << std::endl;
        exit(EXIT_FAILURE);
    }

    cl::Platform platform = pickUpPlatform();
    cl::Device device = pickUpDevice(platform);

    cl::Context runtimeContext({device});
    cl::Program program = loadProgram(runtimeContext, "../kernel.cl");
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

    cl::Kernel simpleAddKernel(program, "simple_add");
    
    int vecSize = atoi(argv[1]);

    std::cout << "# Vector Addition benchmark" << std::endl;
    std::cout << "#N\tTime(OCL)\tTime\t\tMB/s(OCL)\tMB/s" << std::endl;
    for (int i = 1000 ; i < 1e6 ; i = i * 1.5)
    {
        vecSize = i;
        std::vector<int>vec1; vec1.reserve(vecSize);
        std::vector<int>vec2; vec2.reserve(vecSize);
        std::vector<int>vec3(vecSize);

        for (int i = 0; i < vecSize; i++)
        {
            vec1.push_back(i);
            vec2.push_back(i * i);
        }

        cl::Buffer VEC1(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vec1.size() * sizeof(int), vec1.data());
        cl::Buffer VEC2(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vec2.size() * sizeof(int), vec2.data());
        cl::Buffer VEC3(runtimeContext, CL_MEM_WRITE_ONLY, vec3.size() * sizeof(int));

        Timer timer;
        double opencl_time = 0;
        double classic_time = 0;
        for (int j = 0 ; j < 20 ; j++)
        {
            timer.start();
            /* Setting kernel parameters (i.e the ones specified in the associated function in the associated .cl file) */
            simpleAddKernel.setArg(0, VEC1);
            simpleAddKernel.setArg(1, VEC2);
            simpleAddKernel.setArg(2, VEC3);

            /* Creating the command queue that will be used to process */
            cl::CommandQueue queue(runtimeContext, device);
            /* Launch the kernel on the compute device */
            queue.enqueueNDRangeKernel(simpleAddKernel, cl::NullRange, vecSize, cl::NullRange);
            /* Get the result back to host */
            queue.enqueueReadBuffer(VEC3, CL_TRUE, 0, vec3.size() * sizeof(int), vec3.data());
            
            opencl_time += timer.end();
            timer.start();
            vectorAdd(vec1, vec2, &(vec3));
            classic_time += timer.end();
        }
        opencl_time /= 20;
        classic_time /= 20;
        std::cout << vecSize << "\t" 
                    << opencl_time << "\t" 
                    << classic_time << "\t" 
                    << vecSize * sizeof(int) / opencl_time / 1e6 << "\t\t"
                    << vecSize * sizeof(int) / classic_time / 1e6
                    << std::endl;
    }
    
    exit(EXIT_SUCCESS);
}