#include "tensor.h"
#include "tensor_math.h"
#include "mlp.h"
#include <iostream>
#include <cstdlib>

int main() {
    int in = 3, hidden = 4, out = 2;
    MLP mlp(in, hidden, out);

    // init weights
    auto init = [] (Tensor& t) {
        for (size_t i = 0; i < t.data.size(); ++i) {
            t[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    };

    init(mlp.W1); init(mlp.W2);
    // biases can be zero or small
    for (size_t i = 0; i < mlp.b1.data.size(); ++i) mlp.b1[i] = 0.0f;
    for (size_t i = 0; i < mlp.b2.data.size(); ++i) mlp.b2[i] = 0.0f;

    // input: batch 5
    Tensor X({5, in});
    for (size_t i = 0; i < X.data.size(); ++i) {
        X[i] = i * 0.1f;
    }

    Tensor Y = mlp.forward(X);

    // print shape
    std::cout << "Y shape: ";
    for (int s : Y.shape) std::cout << s << " ";
    std::cout << "\n";

    // print first few values
    std::cout << "Y values: \n";
    for (int i = 0; i < Y.shape[0]; ++i) {
        for (int j = 0; j < Y.shape[1]; ++j) {
            std::cout << Y({i,j}) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

