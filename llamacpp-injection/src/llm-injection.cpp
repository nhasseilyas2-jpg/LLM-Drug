// llm-injection core: configuration, hashing, logits site. No ggml dependency (unit-testable).
// Part of LLM Injection Runtime Lab (MIT License).

#include "llm-injection.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>

// ---------------------------------------------------------------------------------------------
// hashing

uint64_t llm_inj_hash(uint64_t a, uint64_t b) {
    // splitmix64 finalizer over a combined key
    uint64_t z = a + 0x9e3779b97f4a7c15ULL * (b + 0x632be59bd9b4e019ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double llm_inj_unit(uint64_t h) {
    return (double) (h >> 11) * (1.0 / 9007199254740992.0); // 2^-53
}

double llm_inj_gauss(uint64_t h) {
    // Box-Muller with both uniforms taken from one 64-bit hash
    const double u1 = ((double) (h >> 32) + 1.0) * (1.0 / 4294967296.0);  // (0, 1]
    const double u2 = (double) (h & 0xffffffffULL) * (1.0 / 4294967296.0); // [0, 1)
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// ---------------------------------------------------------------------------------------------
// configuration

bool llm_inj_config::logits_active() const {
    return logit_noise > 0.0f || logit_temp != 1.0f || top_suppress > 0.0f ||
           (tail_boost > 0.0f && tail_count > 0) || (fixation_bias != 0.0f && !fixation_ids.empty());
}

bool llm_inj_config::graph_active() const {
    return attn_scale != 1.0f || resid_noise > 0.0f || layer_gain != 1.0f || ffn_dropout > 0.0f ||
           ffn_lesion > 0.0f || heads_active() || steer_active();
}

static const char * env_getter(const char * name) {
    const char * v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : nullptr;
}

static void warn_invalid(const char * name, const char * value) {
    std::fprintf(stderr, "llm-injection: ignoring invalid %s='%s'\n", name, value);
}

static float get_float(const llm_inj_getter & get, const char * name, float def, float lo, float hi) {
    const char * v = get(name);
    if (!v) {
        return def;
    }
    char * end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v || !std::isfinite(x)) {
        warn_invalid(name, v);
        return def;
    }
    return (float) std::min<double>(hi, std::max<double>(lo, x));
}

static int64_t get_int(const llm_inj_getter & get, const char * name, int64_t def, int64_t lo, int64_t hi) {
    const char * v = get(name);
    if (!v) {
        return def;
    }
    char * end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    if (end == v) {
        warn_invalid(name, v);
        return def;
    }
    return std::min<int64_t>(hi, std::max<int64_t>(lo, (int64_t) x));
}

static uint64_t get_u64(const llm_inj_getter & get, const char * name, uint64_t def) {
    const char * v = get(name);
    if (!v) {
        return def;
    }
    char * end = nullptr;
    const unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) {
        warn_invalid(name, v);
        return def;
    }
    return (uint64_t) x;
}

bool llm_inj_parse_range(const char * text, llm_inj_range & out) {
    if (text == nullptr) {
        return false;
    }
    std::string s(text);
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
    const size_t colon = s.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    std::string a = s.substr(0, colon);
    std::string b = s.substr(colon + 1);
    const bool abs_a = !a.empty() && (a[0] == 'L' || a[0] == 'l');
    const bool abs_b = !b.empty() && (b[0] == 'L' || b[0] == 'l');
    if (abs_a != abs_b) {
        return false;
    }
    if (abs_a) {
        a = a.substr(1);
        b = b.substr(1);
    }
    if (a.empty() || b.empty()) {
        return false;
    }
    char * end_a = nullptr;
    char * end_b = nullptr;
    const double lo = std::strtod(a.c_str(), &end_a);
    const double hi = std::strtod(b.c_str(), &end_b);
    if (*end_a != '\0' || *end_b != '\0' || !std::isfinite(lo) || !std::isfinite(hi) || lo > hi || lo < 0.0) {
        return false;
    }
    if (!abs_a && hi > 1.0) {
        return false;
    }
    if (abs_a && (lo != std::floor(lo) || hi != std::floor(hi))) {
        return false;
    }
    out.absolute = abs_a;
    out.lo       = (float) lo;
    out.hi       = (float) hi;
    return true;
}

static llm_inj_range get_range(const llm_inj_getter & get, const char * name) {
    llm_inj_range r;
    const char * v = get(name);
    if (v && !llm_inj_parse_range(v, r)) {
        warn_invalid(name, v);
        r = llm_inj_range();
    }
    return r;
}

static std::vector<int32_t> get_ids(const llm_inj_getter & get, const char * name) {
    std::vector<int32_t> ids;
    const char * v = get(name);
    if (!v) {
        return ids;
    }
    const char * p = v;
    while (*p) {
        char * end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p) {
            ++p; // skip separators
            continue;
        }
        if (x >= 0 && x <= INT32_MAX && ids.size() < 4096) {
            ids.push_back((int32_t) x);
        }
        p = end;
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

std::vector<std::pair<int32_t, int32_t>> llm_inj_parse_heads(const char * text) {
    std::vector<std::pair<int32_t, int32_t>> out;
    if (text == nullptr) {
        return out;
    }
    const char * p = text;
    while (*p && out.size() < 4096) {
        char * end = nullptr;
        const long a = std::strtol(p, &end, 10);
        if (end == p) {
            ++p;
            continue;
        }
        p = end;
        if (*p != ':') {
            continue; // a lone number is not a pair
        }
        ++p;
        const long b = std::strtol(p, &end, 10);
        if (end == p) {
            continue;
        }
        p = end;
        if (a >= 0 && a < 65536 && b >= 0 && b < 65536) {
            out.emplace_back((int32_t) a, (int32_t) b);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

llm_inj_config llm_inj_parse(const llm_inj_getter & get) {
    llm_inj_config c;
    c.seed          = get_u64  (get, "LLM_INJ_SEED", 0);
    c.log           = get_int  (get, "LLM_INJ_LOG", 0, 0, 1) != 0;

    c.logit_noise   = get_float(get, "LLM_INJ_LOGIT_NOISE",   0.0f, 0.0f,   50.0f);
    c.logit_temp    = get_float(get, "LLM_INJ_LOGIT_TEMP",    1.0f, 0.05f,  20.0f);
    c.top_suppress  = get_float(get, "LLM_INJ_TOP_SUPPRESS",  0.0f, 0.0f,    1.0f);
    c.tail_boost    = get_float(get, "LLM_INJ_TAIL_BOOST",    0.0f, 0.0f,  100.0f);
    c.tail_count    = (int) get_int(get, "LLM_INJ_TAIL_COUNT", 32, 0, 4096);
    c.fixation_bias = get_float(get, "LLM_INJ_FIXATION_BIAS", 0.0f, -100.0f, 100.0f);
    c.fixation_ids  = get_ids  (get, "LLM_INJ_FIXATION_IDS");

    c.attn_scale    = get_float(get, "LLM_INJ_ATTN_SCALE",    1.0f, 0.01f,  20.0f);
    c.attn_layers   = get_range(get, "LLM_INJ_ATTN_LAYERS");

    c.resid_noise   = get_float(get, "LLM_INJ_RESID_NOISE",   0.0f, 0.0f,   10.0f);
    c.resid_layers  = get_range(get, "LLM_INJ_RESID_LAYERS");
    c.layer_gain    = get_float(get, "LLM_INJ_LAYER_GAIN",    1.0f, -4.0f,   4.0f);
    c.gain_layers   = get_range(get, "LLM_INJ_GAIN_LAYERS");

    c.head_lesion   = get_float(get, "LLM_INJ_HEAD_LESION",   0.0f, 0.0f,    1.0f);
    c.head_gain     = get_float(get, "LLM_INJ_HEAD_GAIN",     0.0f, -4.0f,   4.0f);
    c.head_ids      = llm_inj_parse_heads(get("LLM_INJ_HEAD_IDS"));
    c.head_layers   = get_range(get, "LLM_INJ_HEAD_LAYERS");

    c.ffn_dropout   = get_float(get, "LLM_INJ_FFN_DROPOUT",   0.0f, 0.0f,    0.95f);
    c.ffn_lesion    = get_float(get, "LLM_INJ_FFN_LESION",    0.0f, 0.0f,    1.0f);
    c.ffn_layers    = get_range(get, "LLM_INJ_FFN_LAYERS");

    c.kv_forget     = get_float(get, "LLM_INJ_KV_FORGET",     0.0f, 0.0f,    1.0f);
    c.kv_recent     = (int32_t) get_int(get, "LLM_INJ_KV_RECENT", 32, 0, 1 << 24);
    c.kv_sink       = (int32_t) get_int(get, "LLM_INJ_KV_SINK",    4, 0, 1 << 24);

    if (const char * mode = get("LLM_INJ_NOISE_MODE")) {
        if (std::strcmp(mode, "hash") == 0) {
            c.noise_mode = LLM_INJ_NOISE_HASH;
        } else if (std::strcmp(mode, "sin") != 0) {
            warn_invalid("LLM_INJ_NOISE_MODE", mode);
        }
    }

    c.pk_onset      = get_float(get, "LLM_INJ_PK_ONSET",      0.0f, 0.0f, 100000.0f);
    c.pk_halflife   = get_float(get, "LLM_INJ_PK_HALFLIFE",   0.0f, 0.0f, 100000.0f);

    if (const char * f = get("LLM_INJ_STEER_FILE")) {
        c.steer_file = f;
    }
    c.steer_scale   = get_float(get, "LLM_INJ_STEER_SCALE",   0.0f, -10.0f,   10.0f);
    c.steer_layers  = get_range(get, "LLM_INJ_STEER_LAYERS");
    return c;
}

llm_inj_config llm_inj_parse_env() {
    return llm_inj_parse(env_getter);
}

llm_inj_config llm_inj_parse_text(const std::string & text, const llm_inj_getter & fallback) {
    std::vector<std::pair<std::string, std::string>> kv;
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        while (!line.empty() && (line.back() == '\r' || std::isspace((unsigned char) line.back()))) {
            line.pop_back();
        }
        size_t s = 0;
        while (s < line.size() && std::isspace((unsigned char) line[s])) {
            ++s;
        }
        if (s >= line.size() || line[s] == '#') {
            continue;
        }
        const size_t eq = line.find('=', s);
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = line.substr(s, eq - s);
        while (!key.empty() && std::isspace((unsigned char) key.back())) {
            key.pop_back();
        }
        kv.emplace_back(key, line.substr(eq + 1));
    }
    return llm_inj_parse([&](const char * name) -> const char * {
        for (const auto & p : kv) {
            if (p.first == name) {
                return p.second.empty() ? nullptr : p.second.c_str();
            }
        }
        return fallback ? fallback(name) : nullptr;
    });
}

static std::string range_str(const llm_inj_range & r) {
    char buf[64];
    if (r.absolute) {
        std::snprintf(buf, sizeof(buf), "L%d:L%d", (int) r.lo, (int) r.hi);
    } else {
        std::snprintf(buf, sizeof(buf), "%.3g:%.3g", r.lo, r.hi);
    }
    return buf;
}

std::string llm_inj_describe(const llm_inj_config & c) {
    std::ostringstream os;
    os << "seed=" << c.seed;
    if (c.logits_active()) {
        os << " logits{noise=" << c.logit_noise << " temp=" << c.logit_temp << " top_suppress=" << c.top_suppress
           << " tail=" << c.tail_boost << "x" << c.tail_count << " fixation=" << c.fixation_bias << "x"
           << c.fixation_ids.size() << "}";
    }
    if (c.attn_scale != 1.0f) {
        os << " attention{scale=" << c.attn_scale << " layers=" << range_str(c.attn_layers) << "}";
    }
    if (c.heads_active()) {
        os << " heads{lesion=" << c.head_lesion << " explicit=" << c.head_ids.size() << " gain=" << c.head_gain
           << " layers=" << range_str(c.head_layers) << "}";
    }
    if (c.resid_noise > 0.0f) {
        os << " resid_noise{rel=" << c.resid_noise << " layers=" << range_str(c.resid_layers) << "}";
    }
    if (c.layer_gain != 1.0f) {
        os << " layer_gain{g=" << c.layer_gain << " layers=" << range_str(c.gain_layers) << "}";
    }
    if (c.ffn_dropout > 0.0f) {
        os << " ffn_dropout{p=" << c.ffn_dropout << " layers=" << range_str(c.ffn_layers) << "}";
    }
    if (c.ffn_lesion > 0.0f) {
        os << " ffn_lesion{p=" << c.ffn_lesion << " layers=" << range_str(c.ffn_layers) << "}";
    }
    if (c.kv_active()) {
        os << " kv_forget{p=" << c.kv_forget << " recent=" << c.kv_recent << " sink=" << c.kv_sink << "}";
    }
    if (c.steer_active()) {
        const size_t slash = c.steer_file.find_last_of("/\\");
        os << " steer{scale=" << c.steer_scale << " file=" << (slash == std::string::npos ? c.steer_file : c.steer_file.substr(slash + 1))
           << " layers=" << range_str(c.steer_layers) << "}";
    }
    if ((c.resid_noise > 0.0f || c.ffn_dropout > 0.0f) && c.noise_mode == LLM_INJ_NOISE_HASH) {
        os << " noise=hash";
    }
    if (c.pk_active()) {
        os << " pk{onset=" << c.pk_onset << " halflife=" << c.pk_halflife << "}";
    }
    return os.str();
}

// ---------------------------------------------------------------------------------------------
// process-wide configuration (reloadable)

static std::mutex                                   g_cfg_mutex;
static std::atomic<const llm_inj_config *>          g_cfg{nullptr};
static std::vector<std::unique_ptr<llm_inj_config>> g_cfg_keep; // old configs stay valid (references may be held)
static std::atomic<uint64_t>                        g_generation{0};
static std::string                                  g_cfg_file_text;
static bool                                         g_cfg_file_read = false;
static std::atomic<bool>                            g_fired[LLM_INJ_SITE_COUNT];
static std::atomic<uint64_t>                        g_step{0};

static void announce(const llm_inj_config & c, bool reloadable) {
    if (reloadable) {
        std::fprintf(stderr, "llm-injection: config gen=%llu\n", (unsigned long long) g_generation.load());
    }
    if (c.any_active()) {
        std::fprintf(stderr, "llm-injection: ACTIVE %s\n", llm_inj_describe(c).c_str());
    } else if (c.log || reloadable) {
        std::fprintf(stderr, "llm-injection: inactive (no LLM_INJ_* effect configured)\n");
    }
    std::fflush(stderr);
}

static bool read_file(const char * path, std::string & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    out.clear();
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0 && out.size() < (1u << 20)) {
        out.append(buf, n);
    }
    std::fclose(f);
    return true;
}

// installs a new configuration (caller holds g_cfg_mutex)
static void install(std::unique_ptr<llm_inj_config> c, bool reloadable) {
    const llm_inj_config * p = c.get();
    g_cfg_keep.push_back(std::move(c));
    g_generation.fetch_add(1);
    for (auto & f : g_fired) {
        f.store(false);
    }
    g_cfg.store(p);
    announce(*p, reloadable);
}

// (re)loads the configuration; returns the current one. Caller holds g_cfg_mutex.
static const llm_inj_config * load_locked(bool force) {
    const char * file = env_getter("LLM_INJ_CONFIG_FILE");
    if (file == nullptr) {
        if (g_cfg.load() == nullptr) {
            install(std::make_unique<llm_inj_config>(llm_inj_parse_env()), false);
        }
        return g_cfg.load();
    }
    std::string text;
    if (!read_file(file, text)) {
        text.clear(); // missing file = no intervention beyond the environment
    }
    if (g_cfg.load() == nullptr || force || !g_cfg_file_read || text != g_cfg_file_text) {
        g_cfg_file_text = text;
        g_cfg_file_read = true;
        install(std::make_unique<llm_inj_config>(llm_inj_parse_text(text, env_getter)), true);
    }
    return g_cfg.load();
}

const llm_inj_config & llm_inj_cfg() {
    const llm_inj_config * p = g_cfg.load(std::memory_order_acquire);
    if (p == nullptr) {
        std::lock_guard<std::mutex> lock(g_cfg_mutex);
        p = load_locked(false);
    }
    return *p;
}

uint64_t llm_inj_generation() {
    llm_inj_cfg();
    return g_generation.load();
}

void llm_inj_begin_sequence() {
    {
        std::lock_guard<std::mutex> lock(g_cfg_mutex);
        load_locked(false);
    }
    g_step.store(0);
}

uint64_t llm_inj_step() {
    return g_step.load(std::memory_order_relaxed);
}

float llm_inj_pk_factor(const llm_inj_config & c, double t) {
    if (!c.pk_active()) {
        return 1.0f;
    }
    t = std::max(0.0, t);
    double m = 1.0;
    if (c.pk_onset > 0.0f) {
        m *= 1.0 - std::exp(-std::log(20.0) * t / (double) c.pk_onset); // 95 % absorbed at t = onset
    }
    if (c.pk_halflife > 0.0f) {
        m *= std::pow(0.5, std::max(0.0, t - (double) c.pk_onset) / (double) c.pk_halflife);
    }
    return (float) std::min(1.0, std::max(0.0, m));
}

float llm_inj_pk_now() {
    return llm_inj_pk_factor(llm_inj_cfg(), (double) llm_inj_step());
}

llm_inj_config llm_inj_scaled(const llm_inj_config & c, float m) {
    if (m == 1.0f) {
        return c;
    }
    llm_inj_config s = c;
    s.logit_noise   = c.logit_noise * m;
    s.logit_temp    = 1.0f + (c.logit_temp - 1.0f) * m;
    s.top_suppress  = c.top_suppress * m;
    s.tail_boost    = c.tail_boost * m;
    s.fixation_bias = c.fixation_bias * m;
    s.attn_scale    = 1.0f + (c.attn_scale - 1.0f) * m;
    s.resid_noise   = c.resid_noise * m;
    s.layer_gain    = 1.0f + (c.layer_gain - 1.0f) * m;
    s.head_lesion   = c.head_lesion * m;
    s.head_gain     = 1.0f + (c.head_gain - 1.0f) * m;
    s.ffn_dropout   = c.ffn_dropout * m;
    s.ffn_lesion    = c.ffn_lesion * m;
    s.kv_forget     = c.kv_forget * m;
    s.steer_scale   = c.steer_scale * m;
    return s;
}

static std::atomic<uint64_t> g_built_generation{UINT64_MAX};
static std::atomic<uint64_t> g_built_step{UINT64_MAX};

bool llm_inj_graph_reusable() {
    const llm_inj_config & c = llm_inj_cfg();
    if (g_built_generation.load() != g_generation.load()) {
        return false;
    }
    if (c.pk_active() && c.graph_active() && g_built_step.load() != llm_inj_step()) {
        return false;
    }
    return true;
}

void llm_inj_graph_built() {
    llm_inj_cfg();
    g_built_generation.store(g_generation.load());
    g_built_step.store(llm_inj_step());
}

const char * llm_inj_site_name(llm_inj_site site) {
    switch (site) {
        case LLM_INJ_SITE_LOGITS:      return "logits";
        case LLM_INJ_SITE_ATTENTION:   return "attention";
        case LLM_INJ_SITE_RESID_NOISE: return "resid_noise";
        case LLM_INJ_SITE_LAYER_GAIN:  return "layer_gain";
        case LLM_INJ_SITE_FFN_DROPOUT: return "ffn_dropout";
        case LLM_INJ_SITE_KV_FORGET:   return "kv_forget";
        case LLM_INJ_SITE_HEADS:       return "heads";
        case LLM_INJ_SITE_FFN_LESION:  return "ffn_lesion";
        case LLM_INJ_SITE_STEER:       return "steer";
        default:                       return "unknown";
    }
}

void llm_inj_mark_fired(llm_inj_site site) {
    if (site < 0 || site >= LLM_INJ_SITE_COUNT) {
        return;
    }
    if (g_fired[site].load(std::memory_order_relaxed)) {
        return;
    }
    if (!g_fired[site].exchange(true)) {
        std::fprintf(stderr, "llm-injection: site fired: %s\n", llm_inj_site_name(site));
        std::fflush(stderr);
    }
}

bool llm_inj_has_fired(llm_inj_site site) {
    return site >= 0 && site < LLM_INJ_SITE_COUNT && g_fired[site].load(std::memory_order_relaxed);
}

bool llm_inj_layer_in_range(const llm_inj_range & r, int il, int n_layer) {
    if (il < 0 || n_layer <= 0 || il >= n_layer) {
        return false;
    }
    if (r.absolute) {
        return (float) il >= r.lo && (float) il <= r.hi;
    }
    const float depth = ((float) il + 0.5f) / (float) n_layer;
    return depth >= r.lo && depth <= r.hi;
}

// ---------------------------------------------------------------------------------------------
// logits site

static const uint64_t K_NOISE    = 0x6e6f697365000001ULL;
static const uint64_t K_TAIL     = 0x7461696c00000002ULL;
static const uint64_t K_SUPPRESS = 0x7375707072000003ULL;

void llm_inj_apply_logits_cfg(const llm_inj_config & c, llama_token_data_array * cur_p,
                              uint64_t chain_seed, uint64_t step) {
    if (cur_p == nullptr || cur_p->data == nullptr || cur_p->size == 0 || !c.logits_active()) {
        return;
    }

    llama_token_data * d = cur_p->data;
    const size_t       n = cur_p->size;

    // statistics of the finite logits (masked tokens stay masked)
    double sum = 0.0, sum2 = 0.0;
    size_t nf  = 0;
    for (size_t i = 0; i < n; ++i) {
        const float l = d[i].logit;
        if (std::isfinite(l)) {
            sum  += l;
            sum2 += (double) l * l;
            ++nf;
        }
    }
    if (nf == 0) {
        return;
    }
    const double mean = sum / (double) nf;
    const double var  = sum2 / (double) nf - mean * mean;
    const float  sd   = var > 1e-12 ? (float) std::sqrt(var) : 1.0f;

    const uint64_t base = llm_inj_hash(llm_inj_hash(c.seed, chain_seed), step);

    // 1) gaussian noise, keyed by token id so it does not depend on candidate order
    if (c.logit_noise > 0.0f) {
        const float amp = c.logit_noise * sd;
        const uint64_t kn = llm_inj_hash(base, K_NOISE);
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(d[i].logit)) {
                d[i].logit += amp * (float) llm_inj_gauss(llm_inj_hash(kn, (uint64_t) (uint32_t) d[i].id));
            }
        }
    }

    // 2) tail boost: promote random candidates (re-drawn every step)
    if (c.tail_boost > 0.0f && c.tail_count > 0) {
        const float amp = c.tail_boost * sd;
        const uint64_t kt = llm_inj_hash(base, K_TAIL);
        for (int k = 0; k < c.tail_count; ++k) {
            const size_t idx = (size_t) (llm_inj_hash(kt, (uint64_t) k) % (uint64_t) n);
            if (std::isfinite(d[idx].logit)) {
                d[idx].logit += amp;
            }
        }
    }

    // 3) fixation: persistent bias toward a fixed token set
    if (c.fixation_bias != 0.0f && !c.fixation_ids.empty()) {
        const float amp = c.fixation_bias * sd;
        for (const int32_t id : c.fixation_ids) {
            if ((size_t) id < n && d[id].id == id) {
                // fast path: unsorted full-vocabulary array
                if (std::isfinite(d[id].logit)) {
                    d[id].logit += amp;
                }
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                if (d[i].id == id) {
                    if (std::isfinite(d[i].logit)) {
                        d[i].logit += amp;
                    }
                    break;
                }
            }
        }
    }

    // 4) top-1 suppression: veto the currently preferred token (never the last finite one)
    if (c.top_suppress > 0.0f && nf >= 2 &&
        llm_inj_unit(llm_inj_hash(base, K_SUPPRESS)) < (double) c.top_suppress) {
        size_t imax = n;
        float  lmax = -std::numeric_limits<float>::infinity();
        float  lmin =  std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            const float l = d[i].logit;
            if (std::isfinite(l)) {
                if (l > lmax) { lmax = l; imax = i; }
                if (l < lmin) { lmin = l; }
            }
        }
        if (imax < n) {
            // stays finite so grammar-constrained sampling can still fall back to it
            d[imax].logit = lmin - 4.0f * sd;
        }
    }

    // 5) temperature around the mean (softmax is shift invariant)
    if (c.logit_temp != 1.0f) {
        const float inv_t = 1.0f / c.logit_temp;
        const float m     = (float) mean;
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(d[i].logit)) {
                d[i].logit = m + (d[i].logit - m) * inv_t;
            }
        }
    }

    cur_p->sorted = false;
    llm_inj_mark_fired(LLM_INJ_SITE_LOGITS);
}

void llm_inj_apply_logits(llama_token_data_array * cur_p, uint64_t chain_seed, uint64_t step) {
    const llm_inj_config & c = llm_inj_cfg();
    if (!c.logits_active()) {
        return;
    }
    if (c.pk_active()) {
        llm_inj_apply_logits_cfg(llm_inj_scaled(c, llm_inj_pk_factor(c, (double) step)), cur_p, chain_seed, step);
    } else {
        llm_inj_apply_logits_cfg(c, cur_p, chain_seed, step);
    }
}

void llm_inj_on_sample(llama_token_data_array * cur_p, uint64_t chain_seed, uint64_t step) {
    // the token clock counts sampled tokens; graph sites built for the next token read it
    g_step.store(step + 1, std::memory_order_relaxed);
    llm_inj_apply_logits(cur_p, chain_seed, step);
}

// ---------------------------------------------------------------------------------------------
// lesions and hash noise

static const uint64_t K_HEADS  = 0x6865616473000004ULL;
static const uint64_t K_FFNLES = 0x66666e6c65730005ULL;

float llm_inj_head_gain(const llm_inj_config & c, float m, int il, int n_layer, int head, int n_head) {
    if (!c.heads_active() || head < 0 || head >= n_head || !llm_inj_layer_in_range(c.head_layers, il, n_layer)) {
        return 1.0f;
    }
    bool selected = std::binary_search(c.head_ids.begin(), c.head_ids.end(), std::make_pair((int32_t) il, (int32_t) head));
    const float frac = c.head_lesion * m;
    if (!selected && frac > 0.0f) {
        const int k = (int) std::lround((double) frac * n_head);
        if (k > 0) {
            // rank of this head among the layer's heads by hash (ties broken by index)
            const uint64_t key = llm_inj_hash(c.seed ^ K_HEADS, (uint64_t) (uint32_t) il);
            const uint64_t h   = llm_inj_hash(key, (uint64_t) head);
            int rank = 0;
            for (int j = 0; j < n_head; ++j) {
                const uint64_t hj = llm_inj_hash(key, (uint64_t) j);
                rank += (hj < h || (hj == h && j < head)) ? 1 : 0;
            }
            selected = rank < k;
        }
    }
    return selected ? 1.0f + (c.head_gain - 1.0f) * m : 1.0f;
}

bool llm_inj_ffn_lesioned(const llm_inj_config & c, float m, int il, int64_t ch) {
    const float p = c.ffn_lesion * m;
    if (p <= 0.0f || ch < 0) {
        return false;
    }
    const uint64_t key = llm_inj_hash(c.seed ^ K_FFNLES, (uint64_t) (uint32_t) il);
    return llm_inj_unit(llm_inj_hash(key, (uint64_t) ch)) < (double) p;
}

float llm_inj_hash_noise(uint64_t seed, int il, uint64_t salt, int64_t channel, float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint64_t key = llm_inj_hash(llm_inj_hash(seed, salt), (uint64_t) (uint32_t) il);
    return (float) llm_inj_gauss(llm_inj_hash(key, ((uint64_t) channel << 32) | bits));
}
