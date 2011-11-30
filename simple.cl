
__kernel void test(__global float *input, __global float *output)
{
    const int idx = get_global_id(1) * get_global_size(0) + get_global_id(0);
    output[idx] = input[idx] * 2.0f;
}
