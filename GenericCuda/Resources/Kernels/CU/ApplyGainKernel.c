extern "C" __global__
void ApplyGain(float* data, int size, float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= gain;
    }
}
