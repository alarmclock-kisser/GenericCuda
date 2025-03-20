#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void TestParams(int size, float* array, float value, int duration, float threshold, int silence)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < size)
        {
           // Just an empty test kernel for parameters testing
        }
    }
}