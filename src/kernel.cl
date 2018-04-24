void kernel simple_add(global const int* A, global const int* B, global int* C)
{
    int idx = get_global_id(0);
    C[idx] = A[idx] + B[idx];
}