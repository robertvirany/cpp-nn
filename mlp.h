#pragma once

#include <cassert>
#include <random>

#include "tensor.h"
#include "tensor_math.h"

struct MLP {
  std::vector<Tensor> weights;
  std::vector<Tensor> biases;

  std::vector<Tensor> d_weights;
  std::vector<Tensor> d_biases;

  std::vector<Tensor> Zs;
  std::vector<Tensor> As;

  MLP(int in, std::vector<int> hidden_sizes, int out) {
    assert(!hidden_sizes.empty());

    // input layer to first hidden layer
    weights.push_back({in, hidden_sizes[0]});
    biases.push_back({1, hidden_sizes[0]});

    // hidden layers
    for (size_t i = 0; i < hidden_sizes.size() - 1; ++i) {
      weights.push_back({hidden_sizes[i], hidden_sizes[i + 1]});
      biases.push_back({1, hidden_sizes[i + 1]});
    }

    // last hidden layer to output layer
    weights.push_back({hidden_sizes.back(), out});
    biases.push_back({1, out});
  }

  void init_params() {
    std::mt19937 rng(std::random_device{}());

    for (auto& W : weights) {
      int fan_in = W.shape[0];

      std::normal_distribution<float> dist(
        0.0f,
        std::sqrt(2.0f / fan_in)
      );

      for (size_t i = 0; i < W.numel(); ++i) {
        W[i] = dist(rng);
      }
    }

    for (auto& b : biases) {
      for (size_t i = 0; i < b.numel(); ++i) {
        b[i] = 0.0f;
      }
    }
  }

  Tensor forward(const Tensor& X) {
    As.clear();
    Zs.clear();

    Tensor A = X;
    As.push_back(A);

    for (size_t i = 0; i < weights.size(); ++i) {
      Tensor Z = affine(A, weights[i], biases[i]);
      Zs.push_back(Z);

      A = leaky_relu(Z);
      As.push_back(A);
    }
    return A;
  }

  float mse(const Tensor& X, const Tensor& Y) {
    assert (X.shape == Y.shape);
    float mse = 0.0f;

    for (size_t i = 0; i < X.numel(); ++i) {
      float diff = X[i] - Y[i];
      mse += diff * diff;
    }

    return mse / X.numel();
  }

  Tensor mse_grad(const Tensor& Y_hat, const Tensor& Y) {
    assert(Y_hat.shape == Y.shape);

    Tensor grad(Y_hat.shape);

    float scale = 2.0f / Y_hat.numel();

    for (size_t i = 0; i < Y_hat.numel(); ++i) {
      grad[i] = scale * (Y_hat[i] - Y[i]);
    }

    return grad;
  }

  void backward(const Tensor& Y_hat, const Tensor& Y) {
    assert(Y_hat.shape == Y.shape);

    // dL/dA for final layer
    Tensor dA = mse_grad(Y_hat, Y);

    for (int l = static_cast<int>(weights.size()) - 1; l >= 0; --l) {

      // current layer:
      // Z = A_prev W + b
      // A = leaky_relu(Z)

      Tensor dZ = tensor_matmul(dA, leaky_relu_grad(Zs[l]));

      // gradients wrt params
      d_weights[l] = tensor_matmul(
        transpose(As[l]),
        dZ
      );

      d_biases[l] = sum_rows(dZ);

      // propagate backward unless first layer
      if (l > 0) {
        dA = tensor_matmul(
          dZ,
          transpose(weights[l])
        );
      }
    }
  }

};



