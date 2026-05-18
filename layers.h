#pragma once

#include "tensor.h"
#include "tensor_math.h"
#include <random>

struct Embedding {
  Tensor weight;
  Tensor d_weight;
  std::vector<int> last_tokens;

  Embedding(int vocab, int dim)
      : weight({vocab, dim}), d_weight({vocab, dim}) {}

  void init_params() {
    std::mt19937 rng(std::random_device{}());

    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < weight.numel(); ++i) {
      weight[i] = dist(rng);
    }
  }

  Tensor forward(const std::vector<int> &tokens) {
    last_tokens = tokens;

    int T = tokens.size();
    int C = weight.shape[1];

    Tensor out({T, C});

    for (int t = 0; t < T; ++t) {
      int tok = tokens[t];

      for (int c = 0; c < C; ++c) {
        out({t, c}) = weight({tok, c});
      }
    }

    return out;
  }

  void backward(const Tensor& d_out) {
    int T = d_out.shape[0];
    int C = d_out.shape[1];

    // zero old gradients
    std::fill(
        d_weight.data.begin(),
        d_weight.data.end(),
        0.0f
        );

    // scatter gradients back into embedding rows
    for (int t = 0; t < T; ++t) {
      int tok = last_tokens[t];

      for (int c = 0; c < C; ++c) {
        d_weight({tok, c}) += d_out({t, c});
      }
    }
  }
};

struct Linear {
  Tensor weight;
  Tensor bias;

  Tensor d_weight;
  Tensor d_bias;

  Tensor last_x;

  Linear(int in, int out)
    : weight({in, out}),
    bias({1, out}),
    d_weight({in, out}),
    d_bias({1, out}),
    last_x({0}) {}

  void init_params() {
    std::mt19937 rng(std::random_device{}());

    float stddev = std::sqrt(2.0f / weight.shape[0]);

    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < weight.numel(); ++i) {
      weight[i] = dist(rng);
    }

    for (int i = 0; i < bias.numel(); ++i) {
      bias[i] = 0.0f;
    }
  }

  Tensor forward(const Tensor& x) {
    last_x = x;
    return affine(x, weight, bias);
  }

  Tensor backward(const Tensor& d_out) {
    d_weight = tensor_matmul(
      transpose(last_x),
      d_out
    );

    d_bias = sum_rows(d_out);

    return tensor_matmul(
      d_out,
      transpose(weight)
    );
  }

  void optimize(float lr) {
    for (int i = 0; i < weight.numel(); ++i) {
      weight[i] -= lr * d_weight[i];
    }

    for (int i = 0; i < bias.numel(); ++i) {
      bias[i] -= lr * d_bias[i];
    }
  }
};
