// Exercise Maya's actual fused routing request/response without loading weights.
// CPU polling must see the signal and payload without a driver query or sync.
#include <hip/hip_runtime.h>
#include "strata/kernels/glm_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

int main() {
    namespace gf = strata::kernels::glmf;
    constexpr int E = 64, N = 4096, K = 8, rounds = 100;
    hipDeviceProp_t props{};
    CHECK(hipGetDeviceProperties(&props, 0));
    std::printf("GLM handoff device: %s (%s)\n", props.name, props.gcnArchName);

    // One zeroed device arena; all buffers used by the real routing kernel.
    void* arena = nullptr;
    CHECK(hipMalloc(&arena, 256 << 10));
    CHECK(hipMemset(arena, 0, 256 << 10));
    size_t offset = 0;
    auto take = [&](size_t bytes) {
        offset = (offset + 63) & ~size_t(63);
        void* ptr = (char*) arena + offset;
        offset += bytes;
        return ptr;
    };
    gf::MoeDev d;
    d.n_keys = E;
    d.tab = (unsigned long long*) take((2 * E + gf::kSpares) * 8);
    d.scratch = (unsigned long long*) take(K * 8);
    d.plan_ptr = (unsigned long long*) take(K * 8);
    d.plan_w = (float*) take(K * 4);
    d.plan_id = (int*) take(K * 4);
    d.fetch_src = (unsigned long long*) take(K * 8);
    d.pf_src = (unsigned long long*) take(K * 8);
    d.pf_dst = (unsigned long long*) take(K * 8);
    d.pf_n = (int*) take(4);
    d.seq = (unsigned int*) take(4);
    d.wait_seq = (unsigned int*) take(4);
    d.cpu_seq = (unsigned int*) take(4);
    d.cpu_flag = (int*) take(4);
    d.cpu_part = (float*) take(N * 4);
    float* logits = (float*) take(E * 4);
    float* bias = (float*) take(E * 4);
    float* input = (float*) take(N * 4);
    float* output = (float*) take(N * 4);
    if (offset > (256 << 10)) return 3;

    gf::MoeRequest* ring = nullptr;
    gf::MoeResponse* response = nullptr;
    gf::CpuAnswer* answer = nullptr;
    CHECK(hipHostMalloc((void**) &ring, sizeof(*ring) * gf::kRingSize, hipHostMallocMapped));
    CHECK(hipHostMalloc((void**) &response, sizeof(*response), hipHostMallocMapped));
    CHECK(hipHostMalloc((void**) &answer, sizeof(*answer), hipHostMallocMapped));
    CHECK(hipHostGetDevicePointer(&d.ring, ring, 0));
    void* mapped = nullptr;
    CHECK(hipHostGetDevicePointer(&mapped, response, 0)); d.resp = mapped;
    CHECK(hipHostGetDevicePointer(&mapped, answer, 0)); d.cpu_ans = mapped;

    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    std::vector<float> biases(E), values(N), got(N);
    std::vector<unsigned long long> table(2 * E + gf::kSpares);
    for (int cpu = 0; cpu < 2; ++cpu) {
        std::memset(ring, 0, sizeof(*ring) * gf::kRingSize);
        std::memset(response, 0, sizeof(*response));
        std::memset(answer, 0, sizeof(*answer));
        CHECK(hipMemset(d.seq, 0, 4));
        // Nonzero RAM-tier addresses are identifiers only; this test never fetches
        // or dereferences expert blobs. All eight RAM hits go to the CPU lane.
        for (int e = 0; e < E; ++e) table[E + e] = cpu ? 0x100000000ull + e : 0;
        CHECK(hipMemcpy(d.tab, table.data(), table.size() * 8, hipMemcpyHostToDevice));
        const unsigned long long plan = cpu ? 8ull << 32 : 0;
        hipGraph_t graph;
        hipGraphExec_t exec;
        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        gf::moe_route(logits, bias, E, K, 1.0f, true, 0, input, N, d, nullptr, nullptr, 1.0f, 0, nullptr,
                      stream, nullptr, nullptr, 0, nullptr, nullptr, 0, false, plan);
        gf::moe_wait(d, N, stream);
        gf::moe_cpu_wait(d, N, output, stream);
        CHECK(hipStreamEndCapture(stream, &graph));
        CHECK(hipGraphInstantiateWithFlags(&exec, graph, 0));
        for (int r = 1; r <= rounds; ++r) {
            std::fill(biases.begin(), biases.end(), -10.0f);
            for (int i = 0; i < K; ++i) biases[(r + i) % E] = 8.0f - i;
            for (int e = 0; e < N; ++e) values[e] = (float) (r * 10000 + e);
            CHECK(hipMemcpyAsync(bias, biases.data(), E * 4, hipMemcpyHostToDevice, stream));
            CHECK(hipMemcpyAsync(input, values.data(), N * 4, hipMemcpyHostToDevice, stream));
            CHECK(hipMemsetAsync(output, 0, N * 4, stream));
            CHECK(hipGraphLaunch(exec, stream));
            auto* request = ring + (r % gf::kRingSize);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (__atomic_load_n(&request->seq, __ATOMIC_ACQUIRE) != (unsigned int) r) {
                if (std::chrono::steady_clock::now() > deadline) {
                    std::fprintf(stderr, "GLM request visibility timeout: cpu=%d round=%d\n", cpu, r);
                    return 4;
                }
                std::this_thread::yield();
            }
            if (request->layer != 0 || request->miss_mask != (cpu ? 0u : 255u) ||
                request->cpu_mask != (cpu ? 255u : 0u)) return 5;
            for (int i = 0; i < K; ++i) {
                if (request->ids[i] != (r + i) % E || !std::isfinite(request->w[i]) ||
                    std::fabs(request->w[i] - 0.125f) > 1e-6f) return 6;
                if (cpu && request->cpu_src[i] != 0x100000000ull + (r + i) % E) return 7;
            }
            if (cpu) {
                for (int e = 0; e < N; ++e) if (request->x[e] != values[e]) return 8;
                for (int e = 0; e < N; ++e) answer->part[e] = values[e];
                __atomic_store_n(&answer->seq, (unsigned int) r, __ATOMIC_RELEASE);
            } else {
                response->cpu = 1;
                for (int e = 0; e < N; ++e) response->cpu_part[e] = values[e];
                __atomic_store_n(&response->seq, (unsigned int) r, __ATOMIC_RELEASE);
            }
            CHECK(hipStreamSynchronize(stream)); // only after publishing the host answer
            CHECK(hipMemcpy(got.data(), cpu ? output : d.cpu_part, N * 4, hipMemcpyDeviceToHost));
            for (int e = 0; e < N; ++e) if (got[e] != values[e]) return 9;
        }
        CHECK(hipGraphExecDestroy(exec));
        CHECK(hipGraphDestroy(graph));
    }
    if (gf::launch_errors()) return 10;
    CHECK(hipStreamDestroy(stream));
    CHECK(hipHostFree(answer)); CHECK(hipHostFree(response)); CHECK(hipHostFree(ring)); CHECK(hipFree(arena));
    std::puts("PASS GLM routing graph handoff: 100 disk requests + 100 CPU-lane requests; changing IDs, weights, inputs and answers");
    return 0;
}
