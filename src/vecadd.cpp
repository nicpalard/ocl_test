#define __CL_ENABLE_EXCEPTIONS

#include "opencl_utils.hpp"
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

int main(int argc, char** argv)
{
    cl::Platform platform = get_platform();
    cl::Device device = get_device(platform);

    cl::Context runtimeContext({device});
    cl::Program program = load_and_build_program(runtimeContext, device, "../src/kernelAdd.cl");

    cl::Kernel simpleAddKernel(program, "simple_add");

    std::cout << "# Vector Addition benchmark" << std::endl;
    std::cout << "#N\tTime(OCL)\tTime\t\tMB/s(OCL)\tMB/s" << std::endl;
    for (int i = 1000 ; i < 1e6 ; i = i * 1.5)
    {
        int vecSize = i;
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