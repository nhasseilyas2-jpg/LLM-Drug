// Unit tests for the ggml-free part of llm-injection (config parsing, hashing, logits site, KV forgetting).
// Build: see llamacpp-injection/test/CMakeLists.txt  (or scripts/test-engine.ps1)

#include "llm-injection.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
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

static void test_text_and_heads() {
    // text config: '#' comments, blank lines, CRLF, unknown keys; environment as fallback
    set_env("LLM_INJ_SEED", "77");
    set_env("LLM_INJ_LOGIT_NOISE", "9");
    const std::string text = "# comment\r\nLLM_INJ_LOGIT_NOISE = 0.25\r\n\r\nLLM_INJ_HEAD_IDS=3:1, 0:2;3:1 7\nLLM_INJ_NOISE_MODE=hash\nFOO=bar\n";
    llm_inj_config c = llm_inj_parse_text(text, [](const char * k) -> const char * { return std::getenv(k); });
    CHECK(c.logit_noise == 0.25f);   // file overrides the environment
    CHECK(c.seed == 77ULL);          // environment fallback
    CHECK(c.noise_mode == LLM_INJ_NOISE_HASH);
    CHECK(c.head_ids.size() == 2 && c.head_ids[0] == std::make_pair(0, 2) && c.head_ids[1] == std::make_pair(3, 1));
    CHECK(c.heads_active() && c.graph_active());
    llm_inj_config d = llm_inj_parse_text(text, nullptr);
    CHECK(d.seed == 0ULL);
    set_env("LLM_INJ_SEED", nullptr);
    set_env("LLM_INJ_LOGIT_NOISE", nullptr);

    CHECK(llm_inj_parse_heads(nullptr).empty());
    CHECK(llm_inj_parse_heads("5 6 7").empty());
    CHECK(llm_inj_parse_heads("1:x,2:3").size() == 1);
}

static void test_pk() {
    llm_inj_config c;
    CHECK(!c.pk_active());
    CHECK(llm_inj_pk_factor(c, 0) == 1.0f && llm_inj_pk_factor(c, 1e6) == 1.0f);

    c.pk_onset = 20.0f;
    CHECK(c.pk_active());
    CHECK(llm_inj_pk_factor(c, 0) == 0.0f);
    CHECK(std::fabs(llm_inj_pk_factor(c, 20) - 0.95f) < 1e-4f);
    CHECK(llm_inj_pk_factor(c, 5) < llm_inj_pk_factor(c, 10));
    CHECK(llm_inj_pk_factor(c, 1000) > 0.999f);

    c.pk_halflife = 50.0f;
    CHECK(std::fabs(llm_inj_pk_factor(c, 70) - 0.5 * (1.0 - std::exp(-std::log(20.0) * 3.5))) < 1e-5);
    CHECK(std::fabs(llm_inj_pk_factor(c, 120) - 0.25 * (1.0 - std::exp(-std::log(20.0) * 6.0))) < 1e-5);
    // pure elimination (no onset): halves every half-life
    llm_inj_config e;
    e.pk_halflife = 10.0f;
    CHECK(llm_inj_pk_factor(e, 0) == 1.0f);
    CHECK(std::fabs(llm_inj_pk_factor(e, 10) - 0.5f) < 1e-6f);
    CHECK(std::fabs(llm_inj_pk_factor(e, 30) - 0.125f) < 1e-6f);

    // scaling: additive knobs times m, multiplicative knobs interpolate towards 1
    llm_inj_config s;
    s.logit_noise = 2.0f; s.logit_temp = 3.0f; s.attn_scale = 0.2f; s.layer_gain = -1.0f; s.head_gain = 0.0f;
    s.kv_forget = 0.8f; s.ffn_lesion = 0.5f;
    const llm_inj_config h = llm_inj_scaled(s, 0.5f);
    CHECK(h.logit_noise == 1.0f && h.logit_temp == 2.0f && std::fabs(h.attn_scale - 0.6f) < 1e-6f);
    CHECK(h.layer_gain == 0.0f && h.head_gain == 0.5f && std::fabs(h.kv_forget - 0.4f) < 1e-6f && h.ffn_lesion == 0.25f);
    const llm_inj_config z = llm_inj_scaled(s, 0.0f);
    CHECK(!z.logits_active() && z.attn_scale == 1.0f && z.layer_gain == 1.0f && z.kv_forget == 0.0f);
}

static void test_steer() {
    llm_inj_config c = llm_inj_parse_text("LLM_INJ_STEER_FILE=/tmp/mood.gguf\nLLM_INJ_STEER_SCALE=0.4\nLLM_INJ_STEER_LAYERS=0.25:0.75\n", nullptr);
    CHECK(c.steer_file == "/tmp/mood.gguf" && c.steer_scale == 0.4f && c.steer_active() && c.graph_active());
    CHECK(c.steer_layers.lo == 0.25f && c.steer_layers.hi == 0.75f);
    CHECK(llm_inj_describe(c).find("steer{scale=0.4 file=mood.gguf") != std::string::npos);
    CHECK(llm_inj_scaled(c, 0.5f).steer_scale == 0.2f);
    CHECK(llm_inj_site_name(LLM_INJ_SITE_STEER) == std::string("steer"));

    // a scale without a file (or a file without a scale) does nothing
    CHECK(!llm_inj_parse_text("LLM_INJ_STEER_SCALE=1\n", nullptr).steer_active());
    CHECK(!llm_inj_parse_text("LLM_INJ_STEER_FILE=x.gguf\n", nullptr).steer_active());
    // negative scales are allowed (the opposite pole), the range is clamped
    CHECK(llm_inj_parse_text("LLM_INJ_STEER_FILE=x.gguf\nLLM_INJ_STEER_SCALE=-0.3\n", nullptr).steer_scale == -0.3f);
    CHECK(llm_inj_parse_text("LLM_INJ_STEER_FILE=x.gguf\nLLM_INJ_STEER_SCALE=99\n", nullptr).steer_scale == 10.0f);
}

static void test_head_lesion() {
    llm_inj_config c;
    c.head_lesion = 0.25f; c.seed = 11;
    const int n_head = 32, n_layer = 24;
    int k = 0;
    for (int h = 0; h < n_head; ++h) k += llm_inj_head_gain(c, 1.0f, 5, n_layer, h, n_head) == 0.0f ? 1 : 0;
    CHECK(k == 8);
    // nested in the dose: every head lesioned at m = 0.5 is also lesioned at m = 1
    bool nested = true;
    int k_half = 0;
    for (int h = 0; h < n_head; ++h) {
        const bool half = llm_inj_head_gain(c, 0.5f, 5, n_layer, h, n_head) != 1.0f;
        k_half += half ? 1 : 0;
        nested &= !half || llm_inj_head_gain(c, 1.0f, 5, n_layer, h, n_head) != 1.0f;
    }
    CHECK(nested && k_half == 4);
    // different layers pick different heads
    bool differs = false;
    for (int h = 0; h < n_head; ++h) differs |= (llm_inj_head_gain(c, 1.0f, 5, n_layer, h, n_head) != llm_inj_head_gain(c, 1.0f, 6, n_layer, h, n_head));
    CHECK(differs);
    CHECK(llm_inj_head_gain(c, 0.0f, 5, n_layer, 0, n_head) == 1.0f);
    // partial gain is interpolated by the dose factor
    c.head_gain = 3.0f;
    float g = 1.0f;
    for (int h = 0; h < n_head; ++h) g = std::max(g, llm_inj_head_gain(c, 0.5f, 5, n_layer, h, n_head));
    CHECK(g == 2.0f);
    // explicit heads and layer ranges
    llm_inj_config e;
    e.head_ids = { {2, 3} };
    CHECK(e.heads_active());
    CHECK(llm_inj_head_gain(e, 1.0f, 2, n_layer, 3, n_head) == 0.0f);
    CHECK(llm_inj_head_gain(e, 1.0f, 2, n_layer, 4, n_head) == 1.0f);
    CHECK(llm_inj_head_gain(e, 1.0f, 3, n_layer, 3, n_head) == 1.0f);
    llm_inj_parse_range("L10:L12", e.head_layers);
    CHECK(llm_inj_head_gain(e, 1.0f, 2, n_layer, 3, n_head) == 1.0f);
}

static void test_ffn_lesion_and_hash_noise() {
    llm_inj_config c;
    c.ffn_lesion = 0.3f; c.seed = 5;
    int n = 0, n_half = 0;
    bool nested = true;
    for (int64_t ch = 0; ch < 20000; ++ch) {
        const bool full = llm_inj_ffn_lesioned(c, 1.0f, 4, ch);
        const bool half = llm_inj_ffn_lesioned(c, 0.5f, 4, ch);
        n += full; n_half += half;
        nested &= !half || full;
    }
    CHECK(n > 5700 && n < 6300);
    CHECK(n_half > 2700 && n_half < 3300);
    CHECK(nested);
    CHECK(llm_inj_ffn_lesioned(c, 1.0f, 4, 123) == llm_inj_ffn_lesioned(c, 1.0f, 4, 123));
    CHECK(!llm_inj_ffn_lesioned(c, 0.0f, 4, 123));

    // hash noise: standard normal, independent of the activation value
    double s = 0, s2 = 0, sx = 0;
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        const float x = (float) (i % 1000) * 0.01f;
        const double z = llm_inj_hash_noise(9, 3, 1, i % 64, x);
        s += z; s2 += z * z; sx += z * x;
    }
    const double m = s / N;
    CHECK(std::fabs(m) < 0.02);
    CHECK(std::fabs(s2 / N - m * m - 1.0) < 0.03);
    CHECK(std::fabs(sx / N - m * 4.995) < 0.1); // ~uncorrelated with x
    CHECK(llm_inj_hash_noise(9, 3, 1, 2, 0.5f) == llm_inj_hash_noise(9, 3, 1, 2, 0.5f));
    CHECK(llm_inj_hash_noise(9, 3, 1, 2, 0.5f) != llm_inj_hash_noise(9, 4, 1, 2, 0.5f));
}

static void test_kv_overload() {
    llm_inj_config c;
    c.kv_forget = 1.0f; c.seed = 3;
    CHECK(!llm_inj_kv_forget_pos(c, 0.0f, 100, 1000));
    int a = 0, b = 0;
    for (int p0 = 4; p0 < 4004; ++p0) {
        a += llm_inj_kv_forget_pos(c, 0.25f, p0, 10000) ? 1 : 0;
        b += llm_inj_kv_forget_pos(c, 0.5f, p0, 10000) ? 1 : 0;
    }
    CHECK(a > 900 && a < 1100);
    CHECK(b > 1900 && b < 2100);
}

static void test_reload() {
    const std::string path = "llm-inj-test-config.txt";
    auto write = [&](const char * s) {
        FILE * f = std::fopen(path.c_str(), "wb");
        std::fputs(s, f);
        std::fclose(f);
    };
    write("LLM_INJ_LOGIT_NOISE=0.5\n");
    set_env("LLM_INJ_CONFIG_FILE", path.c_str());
    llm_inj_begin_sequence();
    const uint64_t g0 = llm_inj_generation();
    CHECK(llm_inj_cfg().logit_noise == 0.5f);
    CHECK(!llm_inj_graph_reusable());         // no graph built for this generation yet

    llm_inj_begin_sequence();                 // unchanged file -> same generation
    CHECK(llm_inj_generation() == g0);
    CHECK(llm_inj_step() == 0);

    const llm_inj_config & before = llm_inj_cfg();
    write("LLM_INJ_ATTN_SCALE=2\nLLM_INJ_PK_ONSET=10\n");
    llm_inj_begin_sequence();
    CHECK(llm_inj_generation() == g0 + 1);
    CHECK(llm_inj_cfg().logit_noise == 0.0f && llm_inj_cfg().attn_scale == 2.0f);
    CHECK(before.logit_noise == 0.5f);        // older references stay valid
    CHECK(llm_inj_pk_now() == 0.0f);          // prompt is processed at t = 0

    // graph reuse: invalid after a config change, and every token while a dose schedule is active
    llm_inj_graph_built();
    CHECK(llm_inj_graph_reusable());
    auto v = make_vocab(100);
    auto a = as_array(v);
    llm_inj_on_sample(&a, 1, 0);
    CHECK(llm_inj_step() == 1);
    CHECK(!llm_inj_graph_reusable());
    CHECK(llm_inj_pk_now() > 0.0f);
    llm_inj_graph_built();
    CHECK(llm_inj_graph_reusable());

    std::remove(path.c_str());                // missing file = nothing beyond the environment
    llm_inj_begin_sequence();
    CHECK(llm_inj_generation() == g0 + 2);
    CHECK(!llm_inj_cfg().any_active());
    CHECK(!llm_inj_graph_reusable());
    set_env("LLM_INJ_CONFIG_FILE", nullptr);
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
    test_text_and_heads();
    test_pk();
    test_steer();
    test_head_lesion();
    test_ffn_lesion_and_hash_noise();
    test_kv_overload();
    test_reload();
    std::printf("llm-injection unit tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
