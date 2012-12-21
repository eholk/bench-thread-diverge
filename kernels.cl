// -*- c -*-

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define get(A, N, i, j) ((A)[((i) * (N)) + (j)])

__kernel void MyAdd(const double __global *A,
                    const double __global *B,
                    double __global *C,
                    unsigned int N)
{
    unsigned int i = get_global_id(0);

    for(unsigned int j = 0; j < N; j++) {
        if (i % 2 == 0)
            get(C, N, i, j) = get(A, N, i, j) + get(A, N, i, j);
        else
            get(C, N, i, j) = get(A, N, i, j) - get(A, N, i, j);
    }
}
