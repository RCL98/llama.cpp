#include "common.h"
#include "llama.h"

#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static std::vector<std::string> split_lines(const std::string & s) {
    std::string line;
    std::vector<std::string> lines;
    std::stringstream ss(s);
    while (std::getline(ss, line)) {
        lines.push_back(line);
    }
    return lines;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, int seq_id, bool all_tokens = false) {
    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, { seq_id }, all_tokens || i == tokens.size() - 1);
    }
}

static void mean_pooling(const float * embd, float * out, int n_tokens, int n_embd) {
    for (int i = 0; i < n_embd; i++) {
        out[i] = 0;
    }

    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_embd; j++) {
            out[j] += embd[i * n_embd + j];
        }
    }

    for (int i = 0; i < n_embd; i++) {
        out[i] /= n_tokens;
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        llama_embd_normalize(embd, out, n_embd);
    }
}

static void batch_decode_with_manual_pooling(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }

    float *temp_out = (float *) malloc (batch.n_tokens * n_embd * sizeof(float));
    int token_of_seq = 0;
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = embd = llama_get_embeddings_ith(ctx, i);
        if (embd == NULL) {
            fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, i);
            continue;
        }

        memcpy(temp_out + i * n_embd, embd, n_embd * sizeof(float));
        if (i != 0 && batch.pos[i] == 0) {
            float *seq_tokens = (float *) malloc(token_of_seq * n_embd * sizeof(float));
            memcpy(seq_tokens, temp_out + (i - token_of_seq) * n_embd, token_of_seq * n_embd * sizeof(float));
            mean_pooling(seq_tokens, output + batch.seq_id[i - 1][0] * n_embd, token_of_seq, n_embd);

            token_of_seq = 0;
            free(seq_tokens);
        }
        
        token_of_seq++;
    }
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    params.embedding = true;
    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;

    print_build_info();

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", get_system_info(params).c_str());
    }

    // split the prompt into lines
    std::vector<std::string> prompts = split_lines(params.prompt);

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    for (const auto & prompt : prompts) {
        auto inp = ::llama_tokenize(ctx, prompt, true, false);
        if (inp.size() > n_batch) {
            fprintf(stderr, "%s: error: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        inputs.push_back(inp);
    }

    // add eos if not present
    for (auto & inp : inputs) {
        if (inp.empty() || inp.back() != llama_token_eos(model)) {
            inp.push_back(llama_token_eos(model));
        }
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) inputs.size(); i++) {
            fprintf(stderr, "%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
            for (int j = 0; j < (int) inputs[i].size(); j++) {
                fprintf(stderr, "%6d -> '%s'\n", inputs[i][j], llama_token_to_piece(ctx, inputs[i][j]).c_str());
            }
            fprintf(stderr, "\n\n");
        }
    }

    // initialize batch
    const int n_prompts = prompts.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_prompts * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto & inp = inputs[k];

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + p * n_embd;
            if (params.manual_pooling) {
                batch_decode_with_manual_pooling(ctx, batch, out, s, n_embd);
            } else {
                batch_decode(ctx, batch, out, s, n_embd);
            }
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s, params.manual_pooling);
        s += 1;
    }

    // final batch
    float * out = emb + p * n_embd;
    batch_decode(ctx, batch, out, s, n_embd);

    if (params.logits_file != "") {
        fprintf(stderr, "\nwriting %d embeddings of size %d to %s\n", n_prompts, n_embd, params.logits_file.c_str());
        FILE * f = fopen(params.logits_file.c_str(), "wb");
        if (f) {
            fwrite(emb, sizeof(float), n_prompts * n_embd, f);
            fclose(f);
        }
    } else {
        // print first 3 embeddings
        for (int j = 0; j < std::min(3, n_prompts); j++) {
            fprintf(stderr, "embedding %d: ", j);
            for (int i = 0; i < (int) n_embd; i++) {
                fprintf(stderr, "%f ", emb[j * n_embd + i]);
            }
            fprintf(stderr, "\n\n");
        }
        fprintf(stderr, "\n");
    }

    // clean up
    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
