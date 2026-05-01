#pragma once
#include "tensor.h"
#include "tensor_math.h"

struct MLP {
    Tensor W1, b1;
    Tensor W2, b2;

    MLP(int in, int hidden, int out)
        : W1({in, hidden}), b1({1, hidden}),
          W2({hidden, out}), b2({1, out}) {}

    Tensor forward(const Tensor& X) {
        Tensor Y = leaky_relu(affine(X, W1, b1));
        return affine(Y, W2, b2);
    }
};
