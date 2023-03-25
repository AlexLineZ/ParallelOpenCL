#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <CL\cl.h>
using namespace std;

//большая часть взята отсюда: https://habr.com/ru/post/261323/

cl_kernel formatKernel(string fileName, string kernelName, cl_context& context, cl_device_id& deviceID, cl_int& ret) {

    FILE* fp = fopen(fileName.c_str(), "r");
    char* kernelCode = (char*)malloc(10000);
    size_t kernelSize = fread(kernelCode, 1, 10000, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, (const size_t*)&kernelSize, &ret);
    ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &ret);
    return kernel;
}

unsigned char* negativeFilterWithOpenCL(unsigned char* image, int width, int height) {

    unsigned char* newImage = image;

    cl_uint retNumPlatforms, retNumDevices;
    cl_platform_id platformID;
    cl_device_id deviceID;

    cl_int ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);/* получить доступные платформы */
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices); /* получить доступные устройства */
    cl_context context = clCreateContext(NULL, retNumDevices, &deviceID, NULL, NULL, &ret); /* создать контекст */
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret); /* создаем очередь команд */
    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * 3 * width * height, NULL, &ret);

    cl_kernel kernel = formatKernel("negativeKernel.cl", "negativeFilter", context, deviceID, ret);

    ret = clEnqueueWriteBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * 3 * width * height, newImage, 0, NULL, NULL);/* записать данные в буфер */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer); /* устанавливаем параметр */

    size_t globalWorkSize[1] = { 3 * width * height };
    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * 3 * width * height, newImage, 0, NULL, NULL);

    return newImage;
}


unsigned char* gaussianBlurWithOpenCL(unsigned char* image, int width, int height, int countChannel, int kernelSize, float* gaussKernel)
{
    unsigned char* newImage = image;

    int info[] = { width, height, countChannel, kernelSize };

    cl_uint retNumPlatforms, retNumDevices;
    cl_platform_id platformID;
    cl_device_id deviceID;

    cl_int ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);
    cl_context context = clCreateContext(NULL, retNumDevices, &deviceID, NULL, NULL, &ret);
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret);
    cl_kernel kernel = formatKernel("gaussKernel.cl", "gaussianBlur", context, deviceID, ret);

    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * 3 * width * height, NULL, &ret);
    cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * kernelSize * kernelSize, NULL, &ret);
    cl_mem infoBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 4, NULL, &ret);


    ret = clEnqueueWriteBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * 3 * width * height, newImage, 0, NULL, NULL);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);

    ret = clEnqueueWriteBuffer(commandQueue, kernelBuffer, CL_TRUE, 0, sizeof(float) * kernelSize * kernelSize, gaussKernel, 0, NULL, NULL);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &kernelBuffer);

    ret = clEnqueueWriteBuffer(commandQueue, infoBuffer, CL_TRUE, 0, sizeof(int) * 4, info, 0, NULL, NULL);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &infoBuffer);

    size_t globalWorkSize[1] = { width * height };
    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * 3 * width * height, newImage, 0, NULL, NULL);

    return newImage;
}

unsigned char* convertToThreeChannel(unsigned char* image, int width, int heigth) {
    unsigned char* newImage = new unsigned char[3 * width * heigth];
    int j = 0;
    for (int i = 0; i < width * heigth * 4; i += 4) {
        newImage[j] = image[i];
        newImage[j + 1] = image[i + 1];
        newImage[j + 2] = image[i + 2];
        j += 3;
    }
    return newImage;
}

float* calculateKernel(float sigma, int kernelSize) {
    float* kernel = new float[kernelSize * kernelSize];
    float sum = 0;
    int halfOfKernelSize = kernelSize / 2;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {

            int x = i - halfOfKernelSize;
            int y = j - halfOfKernelSize;

            kernel[i * kernelSize + j] = exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
            sum += kernel[i * kernelSize + j];
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

int main() {

    const char* input = "";
    const char* negative = "negative.png";
    const char* gauss = "gauss.png";

    float* kernel = calculateKernel(7.2, 22);

    int width, height, channels;
    int number;

    cout << "Choose a picture: \n" << "1 - 300x300\n" << "2 - 400x400\n" << "3 - 500x500\n" << "4 - 600x600\n"
        << "5 - 950x950\n" << "6 - 2400x2400\n";

    cin >> number;
    switch (number) {
    case 1:
        input = "300x300.png";
        break;
    case 2:
        input = "400x400.png";
        break;
    case 3:
        input = "500x500.png";
        break;
    case 4:
        input = "600x600.png";
        break;
    case 5:
        input = "950x950.png";
        break;
    case 6:
        input = "2400x2400.png";
        break;
    default:
        input = "image.png";
        break;
    }

    unsigned char* imageData = stbi_load(input, &width, &height, &channels, 0);

    if (imageData == nullptr) {
        cout << "There is no such picture or you entered the wrong name. Try again." << endl;
        return 0;
    }

    if (channels > 3) {
        cout << "Wait. Photo editing in progress..." << endl;
        imageData = convertToThreeChannel(imageData, width, height);
        channels = 3;
    }

    double sum = 0;
    unsigned char* negativeImage;
    unsigned char* gaussImage;


    int filter;
    cout << "Choose a filter: \n" << "1 - OpenCL NegativeFilter\n" << "2 - OpenCL GaussianBlur\n";

    cin >> filter;
    switch (filter) {
    case 1:
        sum = 0;

        for (int i = 0; i < 1000; i++) {

            auto begin = chrono::high_resolution_clock::now();

            negativeImage = negativeFilterWithOpenCL(imageData, width, height);

            auto end = chrono::high_resolution_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);
            cout << elapsedMS.count() / 1000000.0 << endl;
            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(negative, width, height, channels, negativeImage, 0);

        cout << "The middle time of Negative Filter: " << sum / 1000 << " s\n";
        break;

    case 2:
        sum = 0;
        kernel = calculateKernel(7.2, 22);
        for (int i = 0; i < 100; i++) {
            auto begin = chrono::steady_clock::now();

            gaussImage = gaussianBlurWithOpenCL(imageData, width, height, channels, 22, kernel);

            auto end = chrono::steady_clock::now();
            auto elapsedMS = chrono::duration_cast<chrono::microseconds>(end - begin);
            cout << elapsedMS.count() / 1000000.0 << endl;
            sum += elapsedMS.count() / 1000000.0;
        }

        stbi_write_png(gauss, width, height, channels, gaussImage, 0);
        cout << "The middle time of Gauss Filter: " << sum / 100 << " s\n";
        break;

    default:
        cout << "Open another project in Visual Studio 2022" << endl;
        break;
    }

    cout << "Success" << endl;
    stbi_image_free(imageData);
    return 0;
}