// Real batched MLA attention against a CPU softmax reference, without weights.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/kernels/glm_batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

int main() {
    constexpr int T = 5, H = 32, KV = 512, rows = 67, stride = 37;
    const float scale = 1.0f / std::sqrt(128.0f);
    std::vector<float> q(T * H * KV), lat(rows * KV), got(q.size());
    std::vector<uint16_t> half(lat.size());
    std::vector<int> cells(T * stride, -1), counts = {0, 1, 17, 37, 35};
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.4f * std::sin(0.17f * (float) i);
    for (size_t i = 0; i < lat.size(); ++i) {
        const __half h = __float2half(0.7f * std::cos(0.11f * (float) i));
        std::memcpy(&half[i], &h, sizeof(h));
        lat[i] = __half2float(h);
    }
    for (int t = 0; t < T - 1; ++t)
        for (int s = 0; s < counts[t]; ++s)
            cells[t * stride + s] = s % 7 == 5 ? -1 : (11 * s + 3 * t) % rows;
    // The last token has only masked cells; the first has no selected cells.
    float* d_q = nullptr;
    float* d_out = nullptr;
    uint16_t* d_lat = nullptr;
    int* d_cells = nullptr;
    int* d_counts = nullptr;
    CHECK(hipMalloc((void**) &d_q, q.size() * sizeof(float)));
    CHECK(hipMalloc((void**) &d_out, q.size() * sizeof(float)));
    CHECK(hipMalloc((void**) &d_lat, half.size() * sizeof(uint16_t)));
    CHECK(hipMalloc((void**) &d_cells, cells.size() * sizeof(int)));
    CHECK(hipMalloc((void**) &d_counts, counts.size() * sizeof(int)));
    CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_lat, half.data(), half.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_cells, cells.data(), cells.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_counts, counts.data(), counts.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemset(d_out, 0xff, q.size() * sizeof(float))); // catch unwritten outputs
    strata::kernels::glmb::mla_attn(d_q, d_lat, d_cells, d_counts, stride, H, KV, scale, T, d_out, nullptr);
    CHECK(hipDeviceSynchronize());
    if (strata::kernels::glmb::launch_errors()) return 3;
    CHECK(hipMemcpy(got.data(), d_out, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    double worst = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t) t * H + h) * KV;
            std::vector<double> scores(counts[t], -INFINITY);
            double max_score = -INFINITY, denom = 0.0;
            for (int s = 0; s < counts[t]; ++s) {
                const int cell = cells[t * stride + s];
                if (cell < 0) continue;
                double dot = 0.0;
                for (int k = 0; k < KV; ++k) dot += (double) q[base + k] * lat[cell * KV + k];
                scores[s] = dot * scale;
                max_score = std::max(max_score, scores[s]);
            }
            for (double& score : scores) {
                score = std::isfinite(score) ? std::exp(score - max_score) : 0.0;
                denom += score;
            }
            for (int k = 0; k < KV; ++k) {
                double ref = 0.0;
                if (denom > 0.0)
                    for (int s = 0; s < counts[t]; ++s) {
                        const int cell = cells[t * stride + s];
                        if (cell >= 0) ref += scores[s] / denom * lat[cell * KV + k];
                    }
                if (!std::isfinite(got[base + k])) {
                    std::fprintf(stderr, "nonfinite attention output: token=%d head=%d column=%d\n", t, h, k);
                    return 4;
                }
                worst = std::max(worst, std::fabs(got[base + k] - ref));
            }
        }
    }
    CHECK(hipFree(d_counts)); CHECK(hipFree(d_cells)); CHECK(hipFree(d_lat)); CHECK(hipFree(d_out)); CHECK(hipFree(d_q));
    std::printf("GLM prefill attention: max absolute error %.3e (tolerance 5e-5)\n", worst);
    if (worst > 5e-5) return 5;

    // The rocWMMA kernel (FP16 Q) against the F32 kernel on the same FP16-rounded Q: random data, many cells
    {
        constexpr int T2 = 48, H2 = 32, rows2 = 2048, stride2 = 700;
        std::vector<float> q2((size_t) T2 * H2 * KV), lat2((size_t) rows2 * KV), a(q2.size()), b(q2.size());
        std::vector<uint16_t> lat16(lat2.size()), q16(q2.size());
        std::vector<int> cells2((size_t) T2 * stride2, -1), cnt2(T2);
        uint32_t rng = 12345u;
        const auto rnd = [&rng]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xffff) / 32768.0f - 1.0f; };
        const auto to_h = [](float v) { const __half h = __float2half(v); uint16_t u; std::memcpy(&u, &h, 2); return u; };
        for (size_t i = 0; i < q2.size(); ++i) {
            q16[i] = to_h(1.5f * rnd());
            const __half h = *reinterpret_cast<const __half*>(&q16[i]);
            q2[i] = __half2float(h);
        }
        for (size_t i = 0; i < lat2.size(); ++i) lat16[i] = to_h(rnd());
        for (int t = 0; t < T2; ++t) {
            cnt2[t] = t % 5 == 0 ? 3 : std::min(stride2, 40 + 15 * t);   // includes partial 16-cell chunks
            for (int s = 0; s < cnt2[t]; ++s) cells2[(size_t) t * stride2 + s] = (7 * s + 13 * t) % rows2;
        }
        uint16_t *dq16 = nullptr, *dl = nullptr;
        float *dq = nullptr, *da = nullptr, *db = nullptr;
        int *dc = nullptr, *dn = nullptr;
        CHECK(hipMalloc((void**) &dq16, q16.size() * 2)); CHECK(hipMalloc((void**) &dq, q2.size() * 4));
        CHECK(hipMalloc((void**) &dl, lat16.size() * 2)); CHECK(hipMalloc((void**) &da, q2.size() * 4));
        CHECK(hipMalloc((void**) &db, q2.size() * 4)); CHECK(hipMalloc((void**) &dc, cells2.size() * 4));
        CHECK(hipMalloc((void**) &dn, cnt2.size() * 4));
        CHECK(hipMemcpy(dq16, q16.data(), q16.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dq, q2.data(), q2.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dl, lat16.data(), lat16.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dc, cells2.data(), cells2.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dn, cnt2.data(), cnt2.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemset(da, 0xff, q2.size() * 4));
        strata::kernels::glmb::mla_attn_f16q(dq16, dl, dc, dn, stride2, H2, KV, scale, T2, da, nullptr);
        strata::kernels::glmb::mla_attn(dq, dl, dc, dn, stride2, H2, KV, scale, T2, db, nullptr);
        CHECK(hipDeviceSynchronize());
        if (strata::kernels::glmb::launch_errors()) return 3;
        CHECK(hipMemcpy(a.data(), da, a.size() * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), db, b.size() * 4, hipMemcpyDeviceToHost));
        double num = 0, den = 0, worst_head = 0;
        for (size_t r = 0; r < (size_t) T2 * H2; ++r) {
            double rn = 0, rd = 0;
            for (int j = 0; j < KV; ++j) {
                const double d = (double) a[r * KV + j] - b[r * KV + j];
                if (!std::isfinite(a[r * KV + j])) { std::fprintf(stderr, "nonfinite WMMA output row %zu\n", r); return 4; }
                rn += d * d; rd += (double) b[r * KV + j] * b[r * KV + j];
            }
            num += rn; den += rd;
            worst_head = std::max(worst_head, std::sqrt(rn / std::max(1e-30, rd)));
        }
        const double rel = std::sqrt(num / std::max(1e-30, den));
        std::printf("GLM prefill WMMA attention vs F32 kernel: rel L2 %.3e, worst head %.3e (tolerance 1e-3)\n", rel, worst_head);
        if (rel > 1e-3 || worst_head > 5e-3) return 6;
        hipFree(dq16); hipFree(dq); hipFree(dl); hipFree(da); hipFree(db); hipFree(dc); hipFree(dn);
    }
    std::puts("PASS: empty/masked cells, partial tiles, multiple tiles and head groups");
    return 0;
}
