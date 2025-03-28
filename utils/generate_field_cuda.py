import cupy as cp

generate_field_kernel = cp.RawKernel(r'''
extern "C" __global__
void generate_field_kernel(float* X, float* Y, float* r_vals, float* theta_vals, float* output,
                           int num_atoms, int grid_size, float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size) return;

    float x = X[idx];
    float y = Y[idx];
    float R = sqrtf(x * x + y * y);
    float sum_field = 0.0f;

    if (R >= 1e-10f) {
        float r1 = fmaxf(R - delta / 2, 0.0f);
        float r2 = fminf(R + delta / 2, 2 * R);
        float denom = (r2 - r1) * R;

        for (int i = 0; i < num_atoms; i++) {
            float r = r_vals[i];
            float theta = theta_vals[i];

            float C = 0.5f * (3 * cosf(theta) * cosf(theta) - 1);
            float F1 = sqrtf(fmaxf(r * r - r1 * r1, 0.0f));
            float F2 = sqrtf(fmaxf(r * r - r2 * r2, 0.0f));

            sum_field += (1 + C * (3 * (y / r) * (y / r) - 1)) * (F1 - F2) / (denom * r);
        }
    } else {
        for (int i = 0; i < num_atoms; i++) {
            float r = r_vals[i];
            float theta = theta_vals[i];
            
            float C = 0.5f * (3 * cosf(theta) * cosf(theta) - 1);
            
            sum_field += (1 + C * (3 * (y / r) * (y / r) - 1)) / (r * r);
        }
    }
    output[idx] = sum_field;
}
''', 'generate_field_kernel')


def generate_field_cuda(X, Y, r_vals, theta_vals, delta):
    X = cp.array(X, dtype=cp.float32)
    Y = cp.array(Y, dtype=cp.float32)
    r_vals = cp.array(r_vals, dtype=cp.float32)
    theta_vals = cp.array(theta_vals, dtype=cp.float32)
    delta = cp.array(delta, dtype=cp.float32).get()

    grid_size = X.size
    output = cp.zeros_like(X, dtype=cp.float32)

    threads_per_block = 256
    blocks_per_grid = (grid_size + threads_per_block - 1) // threads_per_block

    generate_field_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (X, Y, r_vals, theta_vals, output, r_vals.size, grid_size, delta)
    )
    return output