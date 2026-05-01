#pragma once

#include "tensor.h"
#include <numeric>
#include <vector>
#include <cassert>


inline Tensor tensor_add(const Tensor &a, const Tensor &b) {
    assert(a.shape == b.shape);
    Tensor out(a.shape);

    for (int i = 0; i < a.data.size(); ++i)
        out.data[i] = a.data[i] + b.data[i];

    return out;
}

inline Tensor tensor_hadamard(const Tensor &a, const Tensor &b) {
    assert(a.shape == b.shape);
    Tensor out(a.shape);

    for (int i = 0; i < a.data.size(); ++i)
        out.data[i] = a.data[i] * b.data[i];

    return out;
}

inline Tensor tensor_add_scalar(const Tensor& a, float s) {
    Tensor out(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        out.data[i] = a.data[i] + s;
    }
    return out;
}

inline Tensor tensor_mul_scalar(const Tensor&a, float s) {
    Tensor out(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        out.data[i] = a.data[i] * s;
    }
    return out;
}

inline Tensor reshape(const Tensor&a, const std::vector<int>& new_shape) {
    int n = 1;
    for (int s : new_shape) n *= s;
    assert(n == a.numel());

    Tensor out(new_shape);
    out.data = a.data;
    return out;
}

inline Tensor tensor_matmul(const Tensor &a, const Tensor &b) {
    assert(a.shape.size() == 2 && b.shape.size() == 2);
    assert(a.shape[1] == b.shape[0]);
    Tensor out({a.shape[0], b.shape[1]});

    for (int i = 0; i < a.shape[0]; ++i) {
        for (int j = 0; j < b.shape[1]; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a.shape[1]; ++k) {
                sum += a({i,k}) * b({k,j});
            }
            out({i,j}) = sum;
        }
    }

    return out;
}

inline Tensor affine(const Tensor &X, const Tensor &W, const Tensor &b) {
    Tensor out = tensor_matmul(X, W);

    for (int i = 0; i < out.shape[0]; ++i)
        for (int j = 0; j < out.shape[1]; ++j)
            out ({i,j}) += b({0,j});
    assert(b.shape.size() == 2 && b.shape[1] == out.shape[1]);
    return out;
}

inline Tensor leaky_relu(const Tensor &a) {
    Tensor out(a.shape);

    for (int i = 0; i < out.data.size(); ++i){
        float x = a[i];
        out[i] = x > 0.0f ? x : 0.01f * x;
    }

    return out;
}









