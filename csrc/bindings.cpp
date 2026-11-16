/*
 * PyTorch C++ bindings for Flash Decoding INT4 CUDA kernels
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Must match MAX_DIM_FRAGS * WARP_SIZE in flash_decode_int4.cu: the attention
// kernel holds its output accumulator in 8 lane-register slots per thread.
static const int64_t kMaxHeadDim = 256;

extern "C" {
cudaError_t cu_quantize_int4(const float*, int, int, uint8_t*, float*, float*, cudaStream_t);
cudaError_t cu_flash_decode_int4(const float*, const uint8_t*, const float*, const float*,
                                  const float*, const int*, int, int, int, int, int,
                                  float*, cudaStream_t);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> quantize_int4(torch::Tensor kv) {
    CHECK_INPUT(kv);
    TORCH_CHECK(kv.dim() == 2, "kv must be 2D [num_rows, num_channels]");
    TORCH_CHECK(kv.scalar_type() == torch::kFloat32, "kv must be float32");
    TORCH_CHECK(kv.size(0) > 0 && kv.size(1) > 0,
                "kv must be non-empty, got [", kv.size(0), ", ", kv.size(1), "]");
    TORCH_CHECK(kv.size(0) <= INT32_MAX && kv.size(1) <= INT32_MAX,
                "kv dimensions must fit in int32");

    int num_rows = static_cast<int>(kv.size(0));
    int num_channels = static_cast<int>(kv.size(1));

    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(kv.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(kv.device());

    auto q = torch::empty({num_rows, num_channels}, opts_u8);
    auto scale = torch::empty({num_channels}, opts_f32);
    auto zp = torch::empty({num_channels}, opts_f32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_quantize_int4(
        kv.data_ptr<float>(), num_rows, num_channels,
        q.data_ptr<uint8_t>(), scale.data_ptr<float>(), zp.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "quantize_int4 failed: ", cudaGetErrorString(err));

    return {q, scale, zp};
}

torch::Tensor flash_decode_int4(
    torch::Tensor query,      // [batch, heads, head_dim]
    torch::Tensor k_q,        // [num_blocks, block_size, head_dim]
    torch::Tensor k_scale,    // [num_blocks, head_dim]
    torch::Tensor k_zp,       // [num_blocks, head_dim]
    torch::Tensor value,      // [num_blocks, block_size, head_dim]
    torch::Tensor block_lens  // [num_blocks]
) {
    CHECK_INPUT(query);
    CHECK_INPUT(k_q);
    CHECK_INPUT(k_scale);
    CHECK_INPUT(k_zp);
    CHECK_INPUT(value);
    CHECK_INPUT(block_lens);

    // Every pointer below is handed to device code that indexes it with these
    // sizes. A mismatched shape is an out-of-bounds read on the GPU, which
    // fails silently or corrupts results -- so each one is checked here, where
    // it can still become a readable exception.
    TORCH_CHECK(query.dim() == 3, "query must be 3D [batch, heads, head_dim]");
    TORCH_CHECK(k_q.dim() == 3, "k_q must be 3D [num_blocks, block_size, head_dim]");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(k_q.scalar_type() == torch::kUInt8, "k_q must be uint8");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32, "k_scale must be float32");
    TORCH_CHECK(k_zp.scalar_type() == torch::kFloat32, "k_zp must be float32");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(block_lens.scalar_type() == torch::kInt32, "block_lens must be int32");

    const int64_t hd = query.size(2);
    const int64_t nb = k_q.size(0);
    const int64_t bs = k_q.size(1);

    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0,
                "query must have at least one batch entry and one head");
    TORCH_CHECK(hd >= 1 && hd <= kMaxHeadDim,
                "head_dim must be in [1, ", kMaxHeadDim, "], got ", hd);
    TORCH_CHECK(k_q.size(2) == hd, "k_q last dim must equal head_dim");
    TORCH_CHECK(value.dim() == 3 && value.size(0) == nb && value.size(1) == bs
                && value.size(2) == hd,
                "value must be [num_blocks, block_size, head_dim] matching k_q");
    TORCH_CHECK(k_scale.dim() == 2 && k_scale.size(0) == nb && k_scale.size(1) == hd,
                "k_scale must be [num_blocks, head_dim]");
    TORCH_CHECK(k_zp.dim() == 2 && k_zp.size(0) == nb && k_zp.size(1) == hd,
                "k_zp must be [num_blocks, head_dim]");
    TORCH_CHECK(block_lens.dim() == 1 && block_lens.size(0) == nb,
                "block_lens must be [num_blocks]");

    int batch_size = static_cast<int>(query.size(0));
    int num_heads = static_cast<int>(query.size(1));
    int head_dim = static_cast<int>(hd);
    int num_blocks = static_cast<int>(nb);
    int block_size = static_cast<int>(bs);

    auto output = torch::empty({batch_size, num_heads, head_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_flash_decode_int4(
        query.data_ptr<float>(),
        k_q.data_ptr<uint8_t>(),
        k_scale.data_ptr<float>(),
        k_zp.data_ptr<float>(),
        value.data_ptr<float>(),
        block_lens.data_ptr<int>(),
        batch_size, num_heads, head_dim, num_blocks, block_size,
        output.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "flash_decode_int4 failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4", &quantize_int4, "Per-channel asymmetric INT4 quantization (CUDA)");
    m.def("flash_decode_int4", &flash_decode_int4, "Fused online softmax + INT4 dequant attention (CUDA)");
}
