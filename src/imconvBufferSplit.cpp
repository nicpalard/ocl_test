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

double convolution(cl::CommandQueue queue, 
                cl::Context runtimeContext, 
                cl::Program program,
                float* kernel, 
                size_t k_w, 
                size_t k_h, 
                uchar* im_in, 
                size_t im_w, 
                size_t im_h, 
                uchar* im_out) 
{
    // Create input and output image buffer
    cl::Buffer IMAGE(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, im_w * im_h * sizeof(uchar), im_in);
    cl::Buffer KERNEL(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_w * k_h * sizeof(int), kernel);
    cl::Buffer OUT_IMAGE(runtimeContext, CL_MEM_WRITE_ONLY, im_w * im_h * sizeof(uchar));

    cl::Kernel convKernel(program, "gray_conv_buff");
    convKernel.setArg(0, IMAGE);
    // Kernel scalar arguments can not be of type size_t
    convKernel.setArg(1, (uint) im_w);
    convKernel.setArg(2, (uint) im_h);
    convKernel.setArg(3, KERNEL);
    convKernel.setArg(4, (uint) k_w);
    convKernel.setArg(5, (uint) k_h);
    convKernel.setArg(6, OUT_IMAGE);

    Timer t;
    t.start();
    /* Launch the kernel on the compute device */
    queue.enqueueNDRangeKernel(copyKernel, cl::NullRange, cl::NDRange(im_w, im_h), cl::NullRange);

    im_out = new uchar[im_w * im_h];
    /* Get the result back to host */
    queue.enqueueReadBuffer(OUT_IMAGE, CL_TRUE, 0, im_w * im_h * sizeof(uchar), im_out);
    return t.end();
}

double erosion(cl::CommandQueue queue,
                cl::Context runtimeContext,
                cl::Program program,
                int* structuring_element,
                int se_size,
                uchar* im_in,
                uint im_w,
                uint im_h,
                uchar* im_out)
{
    cl::Buffer IMAGE(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, im_w * im_h * sizeof(uchar), im_in);
    cl::Buffer KERNEL(runtimeContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, se_size * se_size * sizeof(int), structuring_element);
    cl::Buffer OUT_IMAGE(runtimeContext, CL_MEM_WRITE_ONLY, im_w * im_h * sizeof(uchar));

    cl::Kernel erodeKernel(program, "erode");
    erodeKernel.setArg(0, IMAGE);
    erodeKernel.setArg(1, (uint)im_w);
    erodeKernel.setArg(2, (uint)im_h);
    erodeKernel.setArg(3, KERNEL);
    erodeKernel.setArg(4, (uint)se_size);
    erodeKernel.setArg(5, OUT_IMAGE);

    Timer t;
    t.start();
    queue.enqueueNDRangeKernel(erodeKernel, cl::NullRange, cl::NDRange(im_w, im_h), cl::NullRange);
    im_out = new uchar[im_w * im_h];
    queue.enqueueReadBuffer(OUT_IMAGE, CL_TRUE, 0, im_w * im_h * sizeof(uchar), im_out);
    return t.end();
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " src_file.*" << "dst_file.*" << std::endl;
        exit(EXIT_FAILURE);
    }

    cl::Platform platform = get_platform();
    cl::Device device = get_device(platform);

    cl::Context runtimeContext({device});
    cl::Program program = load_and_build_program(runtimeContext, device, "../src/kernelConv.cl");

    /* Creating the command queue that will be used to process */
    cl::CommandQueue queue(runtimeContext, device);

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
    
    size_t k_width = 5;
    size_t k_height = 5;
    float kernel[k_width * k_height] {
        1, -2, -2, -2, 1,
        -1, -2, -3, -2, -1,
        1, -2, -2, -2, 1,
        3, 1, -1, 1, 3
    };
    uchar* im_convolution;
    double convolution_time = convolution(queue, runtimeContext, program, kernel, k_width, k_height, pixels, width, height, im_convolution);


    size_t se_size = 3;
    int structuring_element[se_size * se_size] {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    uchar* im_erosion;
    double erosion_time = erosion(queue, runtimeContext, program, structuring_element, se_size, im_convolution, width, height, im_erosion);

    cv::Mat result;
    cv::cvtColor(cv::Mat(height, width, CV_8UC1, im_erosion), result, CV_GRAY2BGRA);
    cv::imwrite(argv[2], result);

    image.release();
    result.release();
}