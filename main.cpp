#include "mlp.h"
#include "tensor.h"
#include "tensor_math.h"
#include <cstdlib>
#include <iostream>
#include <chrono>

int main() {
  auto start = std::chrono::steady_clock::now();

  int in = 3; std::vector<int> hidden = {3, 4, 5, 6}; int out = 2;
  MLP mlp(in, hidden, out);

  mlp.init_params();

  // input: batch 5
  Tensor X({5, in});
  for (size_t i = 0; i < X.data.size(); ++i) {
    X[i] = i * 0.1f;
  }

  Tensor target({5, out});

  for (size_t i = 0; i < target.numel(); ++i) {
    target[i] = 1.0f;
  }

  for (int step = 0; step < 10000; ++step) {
    Tensor Y = mlp.forward(X);

    float loss = mlp.mse(Y, target);

    mlp.backward(Y, target);
    mlp.optimize(0.01f);

    std::cout << step << " " << loss << "\n";
  }

  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration<double>(end - start);

  std::cout << "wall time: " << elapsed.count() << "\n";

  return 0;
}
