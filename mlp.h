#pragma once

#include <cassert>

#include "tensor.h"
#include "tensor_math.h"

struct MLP {
  std::vector<Tensor> weights;
  std::vector<Tensor> biases;

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

  Tensor forward(const Tensor& X) {
    Tensor Y = X;
    for (size_t i = 0; i < weights.size(); ++i) {
      Y = leaky_relu(affine(Y, weights[i], biases[i]));
    }
    return Y;
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

  Tensor backward(const Tensor& X) {

  }
};
