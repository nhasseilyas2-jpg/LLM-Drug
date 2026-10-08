// Unit tests for the ggml-free part of llm-injection (config parsing, hashing, logits site, KV forgetting).
// Build: see llamacpp-injection/test/CMakeLists.txt  (or scripts/test-engine.ps1)

#include "llm-injection.h"

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_failed = 0;
static int g_passed = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) {                                                                 \
            ++g_passed;                                                             \
        } else {                                                                    \
            ++g_failed;                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                           \
    } while (0)

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static std::vector<llama_token_data> make_vocab(int n) {
    std::vector<llama_token_data> v((size_t) n);
    for (int i = 0; i < n; ++i) {
        // deterministic, peaked distribution: token 7 is the clear favorite
        v[(size_t) i] = { i, (float) std::sin(i * 0.37) * 2.0f + (i == 7 ? 12.0f : 0.0f), 0.0f };
    }
    return v;
}

static llama_token_data_array as_array(std::vector<llama_token_data> & v) {
    return { v.data(), v.size(), -1, false };
}

static int argmax(const std::vector<llama_token_data> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i].logit > v[(size_t) best].logit) best = (int) i;
    }
    return best;
}

static double stddev(const std::vector<llama_token_data> & v) {
    double s = 0, s2 = 0;
    for (const auto & t : v) { s += t.logit; s2 += (double) t.logit * t.logit; }
    const double m = s / v.size();
    return std::sqrt(s2 / v.size() - m * m);
}

static void test_ranges() {
    llm_inj_range r;
    CHECK(llm_inj_parse_range("0.25:0.75", r) && !r.absolute && r.lo == 0.25f && r.hi == 0.75f);
    CHECK(llm_inj_parse_range("L3:L10", r) && r.absolute && r.lo == 3.0f && r.hi == 10.0f);
    CHECK(llm_inj_parse_range(" 0 : 1 ", r) && !r.absolute);
    CHECK(!llm_inj_parse_range("0.8:0.2", r));
    CHECK(!llm_inj_parse_range("0.2:1.5", r));
    CHECK(!llm_inj_parse_range("L3:0.5", r));
    CHECK(!llm_inj_parse_range("abc", r));
    CHECK(!llm_inj_parse_range("L1.5:L2", r));

    llm_inj_range mid; llm_inj_parse_range("0.25:0.75", mid);
    // 24 layers: depth(il) = (il + 0.5) / 24
    CHECK(!llm_inj_layer_in_range(mid, 0, 24));
    CHECK( llm_inj_layer_in_range(mid, 6, 24));   // 0.27
    CHECK( llm_inj_layer_in_range(mid, 17, 24));  // 0.73
    CHECK(!llm_inj_layer_in_range(mid, 18, 24));  // 0.77
    CHECK(!llm_inj_layer_in_range(mid, 24, 24));
    CHECK(!llm_inj_layer_in_range(mid, -1, 24));

    llm_inj_range abs_r; llm_inj_parse_range("L2:L4", abs_r);
    CHECK(!llm_inj_layer_in_range(abs_r, 1, 24));
    CHECK( llm_inj_layer_in_range(abs_r, 2, 24));
    CHECK( llm_inj_layer_in_range(abs_r, 4, 24));
    CHECK(!llm_inj_layer_in_range(abs_r, 5, 24));

    llm_inj_range all;
    int n = 0;
    for (int il = 0; il < 24; ++il) n += llm_inj_layer_in_range(all, il, 24) ? 1 : 0;
    CHECK(n == 24);
}

static void test_env_parsing() {
    const char * keys[] = { "LLM_INJ_SEED", "LLM_INJ_LOGIT_NOISE", "LLM_INJ_LOGIT_TEMP", "LLM_INJ_ATTN_SCALE",
                            "LLM_INJ_ATTN_LAYERS", "LLM_INJ_FFN_DROPOUT", "LLM_INJ_FIXATION_IDS", "LLM_INJ_KV_FORGET",
                            "LLM_INJ_LAYER_GAIN", "LLM_INJ_RESID_NOISE" };
    for (const char * k : keys) set_env(k, nullptr);

    llm_inj_config c0 = llm_inj_parse_env();
    CHECK(!c0.any_active());

    set_env("LLM_INJ_SEED", "12345");
    set_env("LLM_INJ_LOGIT_NOISE", "0.5");
    set_env("LLM_INJ_ATTN_SCALE", "0.5");
    set_env("LLM_INJ_ATTN_LAYERS", "L0:L3");
    set_env("LLM_INJ_FFN_DROPOUT", "3.0");        // clamped to 0.95
    set_env("LLM_INJ_FIXATION_IDS", "5, 9,5;12"); // dedup + sort
    set_env("LLM_INJ_KV_FORGET", "nonsense");     // invalid -> default
    llm_inj_config c = llm_inj_parse_env();
    CHECK(c.seed == 12345ULL);
    CHECK(c.logit_noise == 0.5f);
    CHECK(c.attn_scale == 0.5f && c.attn_layers.absolute && c.attn_layers.hi == 3.0f);
    CHECK(std::fabs(c.ffn_dropout - 0.95f) < 1e-6f);
    CHECK(c.fixation_ids.size() == 3 && c.fixation_ids[0] == 5 && c.fixation_ids[1] == 9 && c.fixation_ids[2] == 12);
    CHECK(c.kv_forget == 0.0f);
    CHECK(c.logits_active() && c.graph_active() && !c.kv_active());
    CHECK(llm_inj_describe(c).find("attention{scale=0.5") != std::string::npos);

    for (const char * k : keys) set_env(k, nullptr);
}

static void test_hash() {
    CHECK(llm_inj_hash(1, 2) == llm_inj_hash(1, 2));
    CHECK(llm_inj_hash(1, 2) != llm_inj_hash(2, 1));
    // gaussian moments over many keys
    double s = 0, s2 = 0;
    const int N = 200000;
    for (int i = 0; i < N; ++i) {
        const double z = llm_inj_gauss(llm_inj_hash(42, (uint64_t) i));
        s += z; s2 += z * z;
    }
    const double m = s / N, var = s2 / N - m * m;
    CHECK(std::fabs(m) < 0.01);
    CHECK(std::fabs(var - 1.0) < 0.02);
    double u = 0;
    for (int i = 0; i < N; ++i) u += llm_inj_unit(llm_inj_hash(7, (uint64_t) i));
    CHECK(std::fabs(u / N - 0.5) < 0.005);
}

static void test_logits_noop() {
    llm_inj_config c;
    auto v = make_vocab(1000);
    const auto ref = v;
    auto a = as_array(v);
    llm_inj_apply_logits_cfg(c, &a, 1, 0);
    bool same = true;
    for (size_t i = 0; i < v.size(); ++i) same &= v[i].logit == ref[i].logit;
    CHECK(same);
}

static void test_logits_noise() {
    llm_inj_config c;
    c.logit_noise = 1.0f;
    c.seed = 99;

    auto v1 = make_vocab(5000), v2 = make_vocab(5000), v3 = make_vocab(5000), v4 = make_vocab(5000);
    const auto ref = make_vocab(5000);
    auto a1 = as_array(v1), a2 = as_array(v2), a3 = as_array(v3), a4 = as_array(v4);
    llm_inj_apply_logits_cfg(c, &a1, 1, 0);
    llm_inj_apply_logits_cfg(c, &a2, 1, 0);  // same seed/step -> identical
    llm_inj_apply_logits_cfg(c, &a3, 1, 1);  // next step -> different
    llm_inj_apply_logits_cfg(c, &a4, 2, 0);  // other trial seed -> different

    bool same12 = true, same13 = true, same14 = true;
    double diff2 = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        same12 &= v1[i].logit == v2[i].logit;
        same13 &= v1[i].logit == v3[i].logit;
        same14 &= v1[i].logit == v4[i].logit;
        diff2 += std::pow(v1[i].logit - ref[i].logit, 2);
    }
    CHECK(same12);
    CHECK(!same13);
    CHECK(!same14);
    // noise RMS ~= 1 std of the original logits
    const double rms = std::sqrt(diff2 / v1.size());
    const double sd = stddev(ref);
    CHECK(rms > 0.9 * sd && rms < 1.1 * sd);
    CHECK(!a1.sorted);

    // noise is keyed by token id, so a permuted array gets the same per-token noise
    auto vp = make_vocab(5000);
    std::vector<llama_token_data> rev(vp.rbegin(), vp.rend());
    auto ar = as_array(rev);
    llm_inj_apply_logits_cfg(c, &ar, 1, 0);
    CHECK(rev[0].id == 4999 && rev[0].logit == v1[4999].logit);
}

static void test_logits_masked_tokens_stay_masked() {
    llm_inj_config c;
    c.logit_noise = 2.0f; c.tail_boost = 50.0f; c.tail_count = 4096; c.logit_temp = 3.0f; c.top_suppress = 1.0f;
    auto v = make_vocab(100);
    for (size_t i = 0; i < v.size(); ++i) if (i != 3 && i != 50) v[i].logit = -INFINITY;
    auto a = as_array(v);
    llm_inj_apply_logits_cfg(c, &a, 1, 0);
    int finite = 0;
    for (const auto & t : v) finite += std::isfinite(t.logit) ? 1 : 0;
    CHECK(finite == 2);

    // a single finite token is never suppressed
    auto w = make_vocab(10);
    for (size_t i = 1; i < w.size(); ++i) w[i].logit = -INFINITY;
    auto b = as_array(w);
    llm_inj_apply_logits_cfg(c, &b, 1, 0);
    CHECK(std::isfinite(w[0].logit));
}

static void test_logits_top_suppress_and_temp() {
    llm_inj_config c;
    c.top_suppress = 1.0f;
    auto v = make_vocab(1000);
    CHECK(argmax(v) == 7);
    auto a = as_array(v);
    llm_inj_apply_logits_cfg(c, &a, 1, 0);
    CHECK(argmax(v) != 7);

    llm_inj_config t;
    t.logit_temp = 2.0f;
    auto w = make_vocab(1000);
    const double sd0 = stddev(w);
    auto b = as_array(w);
    llm_inj_apply_logits_cfg(t, &b, 1, 0);
    CHECK(std::fabs(stddev(w) - sd0 / 2.0) < 1e-3 * sd0);
    CHECK(argmax(w) == 7);

    // suppression probability is honoured approximately
    llm_inj_config h;
    h.top_suppress = 0.3f;
    int hits = 0;
    for (int step = 0; step < 2000; ++step) {
        auto x = make_vocab(50);
        auto ax = as_array(x);
        llm_inj_apply_logits_cfg(h, &ax, 5, (uint64_t) step);
        hits += argmax(x) != 7 ? 1 : 0;
    }
    CHECK(hits > 520 && hits < 680);
}

static void test_logits_fixation() {
    llm_inj_config c;
    c.fixation_bias = 10.0f;
    c.fixation_ids = { 123, 456 };
    auto v = make_vocab(1000);
    const auto ref = v;
    auto a = as_array(v);
    llm_inj_apply_logits_cfg(c, &a, 1, 0);
    const double sd = stddev(ref);
    CHECK(std::fabs((v[123].logit - ref[123].logit) - 10.0 * sd) < 1e-3 * sd);
    CHECK(v[124].logit == ref[124].logit);
    CHECK(argmax(v) == 123 || argmax(v) == 456);

    // sorted / partial candidate arrays use the slow path
    std::vector<llama_token_data> part = { {456, 1.0f, 0.0f}, {7, 5.0f, 0.0f}, {9, 0.0f, 0.0f} };
    auto ap = as_array(part);
    llm_inj_apply_logits_cfg(c, &ap, 1, 0);
    CHECK(part[0].logit > part[1].logit);
}

static void test_kv_forget() {
    llm_inj_config c;
    c.kv_forget = 0.0f;
    CHECK(!llm_inj_kv_forget_pos(c, 100, 1000));

    c.kv_forget = 1.0f; c.kv_recent = 32; c.kv_sink = 4;
    CHECK(!llm_inj_kv_forget_pos(c, 0, 1000));     // sink
    CHECK(!llm_inj_kv_forget_pos(c, 3, 1000));     // sink
    CHECK( llm_inj_kv_forget_pos(c, 4, 1000));
    CHECK(!llm_inj_kv_forget_pos(c, 968, 1000));   // within recent window (distance 32)
    CHECK( llm_inj_kv_forget_pos(c, 967, 1000));
    CHECK(!llm_inj_kv_forget_pos(c, 1000, 1000));  // self

    c.kv_forget = 0.5f; c.seed = 3;
    int hidden = 0;
    for (int p0 = 4; p0 < 10004; ++p0) hidden += llm_inj_kv_forget_pos(c, p0, 20000) ? 1 : 0;
    CHECK(hidden > 4800 && hidden < 5200);
    // the hidden set is consistent across queries (depends only on p0)
    bool consistent = true;
    for (int p0 = 4; p0 < 500; ++p0) consistent &= llm_inj_kv_forget_pos(c, p0, 20000) == llm_inj_kv_forget_pos(c, p0, 30000);
    CHECK(consistent);
}

int main() {
    test_ranges();
    test_env_parsing();
    test_hash();
    test_logits_noop();
    test_logits_noise();
    test_logits_masked_tokens_stay_masked();
    test_logits_top_suppress_and_temp();
    test_logits_fixation();
    test_kv_forget();
    std::printf("llm-injection unit tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
