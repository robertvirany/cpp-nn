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

struct LayerNorm {
  Tensor gamma;
  Tensor beta;

  Tensor d_gamma;
  Tensor d_beta;

  Tensor last_x;
  Tensor last_norm;
  Tensor last_mean;
  Tensor last_inv_std;

  float eps = 1e-5f;

  LayerNorm(int dim)
    : gamma({1, dim}),
      beta({1, dim}),
      d_gamma({1, dim}),
      d_beta({1, dim}),
      last_x({0}),
      last_norm({0}),
      last_mean({0}),
      last_inv_std({0}) {}

  void init_params() {
    for (int i = 0; i < gamma.numel(); ++i) {
      gamma[i] = 1.0f;
      beta[i] = 0.0f;
    }
  }

  Tensor forward(const Tensor& x) {
    assert(x.shape.size() == 2);

    last_x = x;

    int B = x.shape[0];
    int C = x.shape[1];
    last_mean = Tensor({B, 1});
    last_inv_std = Tensor({B, 1});

    Tensor out({B, C});
    last_norm = Tensor({B, C});

    for (int b = 0; b < B; ++b) {
      float mean = 0.0f;

      for (int c = 0; c < C; ++c) {
        mean += x({b,c});
      }

      mean /= C;

      float var = 0.0f;

      for (int c = 0; c < C; ++c) {
        float d = x({b,c}) - mean;
        var += d * d;
      }

      var /= C;

      float inv_std = 1.0f / std::sqrt(var + eps);
      last_mean({b,0}) = mean;
      last_inv_std({b,0}) = inv_std;

      for (int c = 0; c < C; ++c) {
        float norm = (x({b,c}) - mean) * inv_std;

        last_norm({b,c}) = norm;

        out({b,c}) = 
          norm * gamma({0,c}) + beta({0,c});
      }
    }

    return out;
  }

  Tensor backward(const Tensor& d_out) {
    int B = d_out.shape[0];
    int C = d_out.shape[1];

    Tensor d_x({B, C});

    std::fill(d_gamma.data.begin(), d_gamma.data.end(), 0.0f);
    std::fill(d_beta.data.begin(), d_beta.data.end(), 0.0f);

    for (int b = 0; b < B; ++b) {
      float mean_dy = 0.0f;
      float mean_dy_xhat = 0.0f;

      for (int c = 0; c < C; ++c) {
        float dy = d_out({b,c});
        float xhat = last_norm({b,c});

        mean_dy += dy;
        mean_dy_xhat += dy * xhat;

        // Layernorm backprop formulas
        d_gamma({0,c}) += dy * xhat;
        d_beta({0,c}) += dy;
      }

      mean_dy /= C;
      mean_dy_xhat /= C;

      float inv_std = last_inv_std({b,0});

      for (int c = 0; c < C; ++c) {
        float dy = d_out({b,c});
        float xhat = last_norm({b,c});

        d_x({b,c}) =
          gamma({0,c}) * inv_std *
          (dy - mean_dy - xhat * mean_dy_xhat);
      }
    }

    return d_x;
  }

  void optimize(float lr) {
    for (int i = 0; i < gamma.numel(); ++i) {
      gamma[i] -= lr * d_gamma[i];
    }

    for (int i = 0; i < beta.numel(); ++i) {
      beta[i] -= lr * d_beta[i];
    }
  }
};
