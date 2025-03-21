extern "C" __global__
void ApplyClipping(float* data, int size, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(minVal, fminf(data[idx], maxVal));
    }
}