// Low-Pass Filter mit einfachem rekursiven RC-Filter
extern "C" __global__
void LowPassFilter(float* data, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < size) {
        data[idx] = alpha * data[idx] + (1 - alpha) * data[idx - 1];
    }
}