/*
 * Flash Decoding with INT4 KV: CUDA Kernels
 *
 * Implements:
 * - Per-channel asymmetric INT4 quantization / dequantization
 * - Fused online softmax (Welford's algorithm) over paged blocks
 * - On-the-fly INT4 dequant in shared memory (no materialization)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstddef>
#include <cstdint>

#define WARP_SIZE 32

/*
 * quantize_int4_kernel
 *
 * Per-channel asymmetric INT4 quantization.
 * Grid: (num_channels), Block: (256 threads over rows)
 *
 * scale[c] = (max[c] - min[c]) / 15
 * q = clip(round((kv - min[c]) / scale[c]), 0, 15)
 */
__global__ void quantize_int4_kernel(
    const float* kv,        // [num_rows, num_channels]
    int num_rows,
    int num_channels,
    uint8_t* q_out,         // [num_rows, num_channels] (INT4 stored per byte)
    float* scale_out,       // [num_channels]
    float* zp_out           // [num_channels]
) {
    int c = blockIdx.x;
    if (c >= num_channels) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* s_min = smem;
    float* s_max = smem + blockDim.x;

    // Find min/max over rows for this channel
    float local_min = CUDART_INF_F;
    float local_max = -CUDART_INF_F;
    for (int r = tid; r < num_rows; r += blockDim.x) {
        float v = kv[(size_t)r * num_channels + c];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float min_val = s_min[0];
    float max_val = s_max[0];
    float range = fmaxf(max_val - min_val, 1e-8f);
    float scale = range / 15.0f;

    if (tid == 0) {
        scale_out[c] = scale;
        zp_out[c] = -min_val / scale;
    }
    __syncthreads();

    // Quantize
    for (int r = tid; r < num_rows; r += blockDim.x) {
        float v = kv[(size_t)r * num_channels + c];
        float qf = (v - min_val) / scale;
        int qi = (int)(qf + 0.5f);
        qi = max(0, min(15, qi));
        q_out[(size_t)r * num_channels + c] = (uint8_t)qi;
    }
}

/*
 * flash_decode_int4_kernel
 *
 * Fused online softmax over INT4-quantized paged KV blocks.
 * Grid: (batch_size, num_heads)
 * Block: (128 threads = 4 warps)
 *
 * Parallelization (flash-decoding style):
 *   - Warps stride over sequence positions; each warp keeps its own
 *     online-softmax state (m, l) plus an output accumulator whose
 *     head_dim slices live in lane registers (lane d, d+32, ...).
 *   - Per position, lanes compute a partial q·dequant(k) dot product,
 *     reduced with warp shuffles — no block-wide syncs in the hot loop.
 *   - Per-page scales/zero-points are staged in shared memory once per
 *     page (zp pre-multiplied by scale so dequant is one FMA).
 *   - Warps merge at the end with a standard log-sum-exp combine.
 */
#define MAX_DIM_FRAGS 8   // supports head_dim up to 8*32 = 256

__global__ void flash_decode_int4_kernel(
    const float* query,       // [batch, heads, head_dim]
    const uint8_t* k_q,       // [num_blocks, block_size, head_dim] INT4
    const float* k_scale,     // [num_blocks, head_dim]
    const float* k_zp,        // [num_blocks, head_dim]
    const float* value,       // [num_blocks, block_size, head_dim] FP32
    const int* block_lens,    // [num_blocks]
    int batch_size,
    int num_heads,
    int head_dim,
    int num_blocks,
    int block_size,
    float* output             // [batch, heads, head_dim]
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= batch_size || h >= num_heads) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    extern __shared__ float smem[];
    float* q_vec   = smem;                          // head_dim
    float* s_scale = q_vec + head_dim;              // head_dim
    float* s_zps   = s_scale + head_dim;            // head_dim (zp * scale)
    float* s_m     = s_zps + head_dim;              // num_warps
    float* s_l     = s_m + num_warps;               // num_warps
    float* s_acc   = s_l + num_warps;               // num_warps * head_dim

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_vec[d] = query[(b * num_heads + h) * head_dim + d];
    }

    // Per-warp online-softmax state; accumulator distributed over lanes
    float m_w = -CUDART_INF_F;
    float l_w = 0.0f;
    float acc[MAX_DIM_FRAGS];
    #pragma unroll
    for (int i = 0; i < MAX_DIM_FRAGS; i++) acc[i] = 0.0f;

    for (int blk = 0; blk < num_blocks; blk++) {
        __syncthreads();   // previous page's smem scales no longer in use
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float sc = k_scale[blk * head_dim + d];
            s_scale[d] = sc;
            s_zps[d] = k_zp[blk * head_dim + d] * sc;
        }
        __syncthreads();

        // Defence in depth: the Python layer validates lengths, but a length
        // outside [0, block_size] must never become an out-of-bounds read.
        const int blen = min(max(block_lens[blk], 0), block_size);
        for (int s = warp; s < blen; s += num_warps) {
            const size_t row = ((size_t)blk * block_size + s) * head_dim;

            // Lane-partial dot product, then warp-shuffle reduction
            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += 32) {
                float kd = (float)k_q[row + d] * s_scale[d] - s_zps[d];
                partial += q_vec[d] * kd;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, off);
            }
            const float logit = __shfl_sync(0xffffffff, partial, 0);

            // Online update — identical scalars in every lane of the warp
            const float m_new = fmaxf(m_w, logit);
            const float corr = expf(m_w - m_new);
            const float p = expf(logit - m_new);
            int i = 0;
            for (int d = lane; d < head_dim; d += 32, i++) {
                acc[i] = acc[i] * corr + p * value[row + d];
            }
            l_w = l_w * corr + p;
            m_w = m_new;
        }
    }

    // Publish per-warp state, then log-sum-exp merge across warps
    if (lane == 0) {
        s_m[warp] = m_w;
        s_l[warp] = l_w;
    }
    {
        int i = 0;
        for (int d = lane; d < head_dim; d += 32, i++) {
            s_acc[warp * head_dim + d] = acc[i];
        }
    }
    __syncthreads();

    float m_star = -CUDART_INF_F;
    for (int w = 0; w < num_warps; w++) m_star = fmaxf(m_star, s_m[w]);

    float l_star = 0.0f;
    for (int w = 0; w < num_warps; w++) {
        // A warp that saw no positions has m = -inf; exp(-inf - m_star)
        // would be NaN when m_star is also -inf, so gate explicitly.
        if (s_m[w] != -CUDART_INF_F) {
            l_star += s_l[w] * expf(s_m[w] - m_star);
        }
    }
    l_star = fmaxf(l_star, 1e-10f);

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            if (s_m[w] != -CUDART_INF_F) {
                o += s_acc[w * head_dim + d] * expf(s_m[w] - m_star);
            }
        }
        output[(b * num_heads + h) * head_dim + d] = o / l_star;
    }
}

/*
 * Wrapper functions for CPU interface
 */
extern "C" {

cudaError_t cu_quantize_int4(
    const float* kv,
    int num_rows,
    int num_channels,
    uint8_t* q_out,
    float* scale_out,
    float* zp_out,
    cudaStream_t stream
) {
    if (num_rows <= 0 || num_channels <= 0) return cudaErrorInvalidValue;

    int threads = 256;
    int blocks = num_channels;
    int shared = 2 * threads * sizeof(float);

    quantize_int4_kernel<<<blocks, threads, shared, stream>>>(
        kv, num_rows, num_channels, q_out, scale_out, zp_out
    );
    return cudaGetLastError();
}

cudaError_t cu_flash_decode_int4(
    const float* query,
    const uint8_t* k_q,
    const float* k_scale,
    const float* k_zp,
    const float* value,
    const int* block_lens,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_blocks,
    int block_size,
    float* output,
    cudaStream_t stream
) {
    // The kernel keeps its output accumulator in MAX_DIM_FRAGS lane-register
    // slots per thread. A wider head_dim would index past that array -- an
    // out-of-bounds write in device code -- so it is rejected here.
    if (head_dim <= 0 || head_dim > MAX_DIM_FRAGS * WARP_SIZE) return cudaErrorInvalidValue;
    if (batch_size <= 0 || num_heads <= 0 || num_blocks < 0 || block_size < 0)
        return cudaErrorInvalidValue;

    dim3 grid(batch_size, num_heads);
    int threads = 128;
    int num_warps = threads / 32;
    // q_vec + s_scale + s_zps + s_m + s_l + s_acc
    int shared = (3 * head_dim + 2 * num_warps + num_warps * head_dim)
                 * (int)sizeof(float);

    flash_decode_int4_kernel<<<grid, threads, shared, stream>>>(
        query, k_q, k_scale, k_zp, value, block_lens,
        batch_size, num_heads, head_dim, num_blocks, block_size,
        output
    );
    return cudaGetLastError();
}

} // extern "C"
