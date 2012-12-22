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

__kernel void MyAdd_col(const double __global *A,
                        const double __global *B,
                        double __global *C,
                        unsigned int N)
{
    unsigned int i = get_global_id(0);

    for(unsigned int j = 0; j < N; j++) {
        if (j % 2 == 0)
            get(C, N, i, j) = get(A, N, i, j) + get(A, N, i, j);
        else
            get(C, N, i, j) = get(A, N, i, j) - get(A, N, i, j);
    }
}

__kernel void MyAdd_2D(const double __global *A,
                       const double __global *B,
                       double __global *C,
                       unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i % 2 == 0)
        get(C, N, i, j) = get(A, N, i, j) + get(A, N, i, j);
    else
        get(C, N, i, j) = get(A, N, i, j) - get(A, N, i, j);
}

__kernel void MyAdd_2D_col(const double __global *A,
                           const double __global *B,
                           double __global *C,
                           unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (j % 2 == 0)
        get(C, N, i, j) = get(A, N, i, j) + get(A, N, i, j);
    else
        get(C, N, i, j) = get(A, N, i, j) - get(A, N, i, j);
}

__kernel void MyAdd_2D_unweave(const double __global *A,
                               const double __global *B,
                               double __global *C,
                               unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if(i < N / 2) {
        unsigned int i = 2 * i;
        get(C, N, i, j) = get(A, N, i, j) + get(A, N, i, j);
    }
    else {
        unsigned int i = 2 * i + 1;
        get(C, N, i, j) = get(A, N, i, j) - get(A, N, i, j);
    }
}

__kernel void MyAdd_2D_unweave_col(const double __global *A,
                                   const double __global *B,
                                   double __global *C,
                                   unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if(j < N / 2) {
        unsigned int j = 2 * j;
        get(C, N, i, j) = get(A, N, i, j) + get(A, N, i, j);
    }
    else {
        unsigned int j = 2 * j + 1;
        get(C, N, i, j) = get(A, N, i, j) - get(A, N, i, j);
    }
}
