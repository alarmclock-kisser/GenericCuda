// High-Pass Filter mit einfachem rekursiven RC-Filter
extern "C" __global__
void HighPassFilter(float* data, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < size) {
        data[idx] = alpha * (data[idx] - data[idx - 1]) + data[idx - 1];
    }
}