__kernel void negativeFilter(__global unsigned char* newImage)
{
    int gid = get_global_id(0);
    newImage[gid] = 255 - newImage[gid];
}


   