#include <opencv2/opencv.hpp>

#include "opencl_utils.hpp"
#include "benchmark.hpp"

float* create_gaussian_kernel(float sigma, size_t kernel_size)
{
    float* kernel = new float[kernel_size * kernel_size];
    float mean = kernel_size/2;
    float sum = 0.0;
    for (int i = 0; i < kernel_size * kernel_size; ++i) 
    {
        int x = i % kernel_size;
        int y = (i - x) / kernel_size % kernel_size;
        kernel[i] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) ) / (2 * M_PI * sigma * sigma);
        // Accumulate the kernel values
        sum += kernel[i];
    }

    // Normalize the kernel
    
    /*
    for (int i = 0; i < kernel_size * kernel_size; ++i)
            kernel[i] /= sum;
    */
    return kernel;
}

int main(int argc, char** argv)
{
    size_t k_width = 5;
    size_t k_height = 5;
    //float* kernel = create_gaussian_kernel(0.8, k_width);
    float kernel[k_width * k_height] {
         3,   1, -1,  1,  3, 
         1,  -2, -2, -2,  1,
        -1,  -2, -3, -2, -1,
         1,  -2, -2, -2,  1,
         3,   1, -1,  1,  3
    };

    size_t se_size = 3;
    int structuring_element[se_size * se_size] {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

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
    cv::cvtColor(image, image_rgba, CV_BGRA2GRAY);
    
    // Get pixel data array
    int size = strlen(reinterpret_cast<char*>(image_rgba.data));
    uchar* pixels = new uchar[width * height];
    pixels = image_rgba.data;

    cl::Platform platform = get_platform();
    cl::Device device = get_device(platform);

    cl::Context runtimeContext({device});
    cl::Program program = load_and_build_program(runtimeContext, device, "../src/kernelConv.cl");

    Timer t;
    t.start();
    // Create input and output image buffer
    cl::Buffer IMAGE(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(uchar), pixels);
    cl::Buffer KERNEL(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_width * k_height * sizeof(int), kernel);
    cl::Buffer OUT_IMAGE(runtimeContext, CL_MEM_WRITE_ONLY, width * height * sizeof(uchar));
    
    cl::Kernel convKernel(program, "gray_conv_buff");
    convKernel.setArg(0, IMAGE);
    // Kernel scalar arguments can not be of type size_t
    convKernel.setArg(1, (uint) width);
    convKernel.setArg(2, (uint) height);
    convKernel.setArg(3, KERNEL);
    convKernel.setArg(4, (uint) k_width);
    convKernel.setArg(5, (uint) k_height);
    convKernel.setArg(6, OUT_IMAGE);

    /* Creating the command queue that will be used to process */
    cl::CommandQueue queue(runtimeContext, device);
    /* Launch the kernel on the compute device */
    queue.enqueueNDRangeKernel(convKernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);

    uchar* copy_pixels = new uchar[width * height];
    /* Get the result back to host */
    queue.enqueueReadBuffer(OUT_IMAGE, CL_TRUE, 0, width * height * sizeof(uchar), copy_pixels);

    IMAGE = cl::Buffer(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(uchar), copy_pixels);
    KERNEL = cl::Buffer(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, se_size * se_size * sizeof(float), structuring_element);
    OUT_IMAGE = cl::Buffer(runtimeContext, CL_MEM_WRITE_ONLY, width * height * sizeof(uchar));

    cl::Kernel erodeKernel(program, "erode");
    erodeKernel.setArg(0, IMAGE);
    erodeKernel.setArg(1, (uint)width);
    erodeKernel.setArg(2, (uint)height);
    erodeKernel.setArg(3, KERNEL);
    erodeKernel.setArg(4, (uint)se_size);
    erodeKernel.setArg(5, OUT_IMAGE);

    queue.enqueueNDRangeKernel(erodeKernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    uchar* erode_pixels = new uchar[width * height];
    queue.enqueueReadBuffer(OUT_IMAGE, CL_TRUE, 0, width * height * sizeof(uchar), erode_pixels);
    float end_time = t.end();

    std::cout << "Convolution & erosion done in " << end_time << std::endl;

    cv::Mat result;
    cv::cvtColor(cv::Mat(height, width, CV_8UC1, erode_pixels), result, CV_GRAY2BGRA);
    cv::imwrite(argv[2], result);

    image.release();
    result.release();
}