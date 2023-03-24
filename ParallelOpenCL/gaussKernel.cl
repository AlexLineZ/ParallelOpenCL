__kernel void gaussianBlur(__global unsigned char* newImage, __global float* gaussKernel, __global int* info) {

    int gid = get_global_id(0);

    int width = info[0];
    int height = info[1];
    int countChannel = info[2];
    int kernelSize = info[3];

    float pixelR = 0, pixelG = 0, pixelB = 0;

    int y = gid / width;
    int x = gid % width;

    int halfOfKernelSize = kernelSize / 2;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            
            int pixelX = x + i - halfOfKernelSize;
            int pixelY = y + j - halfOfKernelSize;

            if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {

                int ind = (pixelY * width + pixelX) * countChannel;

                pixelR += gaussKernel[i * kernelSize + j] * newImage[ind];
                pixelG += gaussKernel[i * kernelSize + j] * newImage[ind + 1];
                pixelB += gaussKernel[i * kernelSize + j] * newImage[ind + 2];
            }

        }
    }
    int index = (y * width + x) * countChannel;

    newImage[index] = pixelR;
    newImage[index + 1] = pixelG;
    newImage[index + 2] = pixelB;
}
