#include <cassert>
#include <iostream>
#include <vector>

struct matrix {
  int rows;
  int cols;
  std::vector<float> data;

  matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

  float &operator()(int r, int c) { return data[r * cols + c]; }
  float operator()(int r, int c) const { return data[r * cols + c]; }
};

matrix matmul(const matrix &a, const matrix &b) {
  assert(a.cols == b.rows);

  matrix out(a.rows, b.cols);
  for (int row = 0; row < a.rows; ++row) {
    for (int col = 0; col < b.cols; ++col) {
      float sum = 0.0f;

      for (int k = 0; k < a.cols; ++k) {
        sum += a(row, k) * b(k, col);
      }

      out(row, col) = sum;
    }
  }

  return out;
}

matrix matadd(const matrix &a, const matrix &b) {
  assert(a.rows == b.rows && a.cols == b.cols);

  matrix out(a.rows, a.cols);
  for (int row = 0; row < a.rows; ++row) {
    for (int col = 0; col < a.cols; ++col) {

      out(row, col) = a(row, col) + b(row, col);
    }
  }

  return out;
}

matrix transpose(const matrix &a) {

  matrix out(a.cols, a.rows);
  for (int row = 0; row < a.rows; ++row) {
    for (int col = 0; col < a.cols; ++col) {

      out(col, row) = a(row, col);
    }
  }

  return out;
}

matrix hadamard(const matrix &a, const matrix &b) {
  assert(a.rows == b.rows && a.cols == b.cols);

  matrix out(a.rows, a.cols);
  for (int row = 0; row < a.rows; ++row) {
    for (int col = 0; col < a.cols; ++col) {
      out(row, col) = a(row, col) * b(row, col);
    }
  }

  return out;
}

matrix scalar_mul(const matrix &a, float s) {

  matrix out(a.rows, a.cols);
  for (int row = 0; row < a.rows; ++row) {
    for (int col = 0; col < a.cols; ++col) {
      out(row, col) = s * a(row, col);
    }
  }

  return out;
}

matrix mat_sub(const matrix &a, const matrix &b) {
  assert(a.rows == b.rows && a.cols == b.cols);

  matrix minus_b = scalar_mul(b, -1);

  matrix out = matadd(a, minus_b);

  return out;
}

matrix affine(const matrix &X, const matrix &W, const matrix &b) {
  matrix out = matmul(X, W);
  out = matadd(out, b);

  return out;
}

float mean_squared_error(const matrix &Y, const matrix &Y_hat) {
  assert(Y.rows == Y_hat.rows && Y.cols == Y_hat.cols);
  int n = Y.rows;
  int d = Y.cols;

  float mse = 0.0f;
  for (int row = 0; row < Y.rows; ++row) {
    for (int col = 0; col < Y.cols; ++col) {
      float diff = Y(row, col) - Y_hat(row, col);

      mse += diff * diff;
    }
  }

  mse /= static_cast<float>(n * d);

  return mse;
}

matrix mse_grad(const matrix &Y, const matrix &Y_hat) {
  assert(Y.rows == Y_hat.rows && Y.cols == Y_hat.cols);
  int n = Y.rows;
  int d = Y.cols;
  int N = n * d;

  matrix grad(n, d);

  float scale = 2.0f / static_cast<float>(N);

  for (int row = 0; row < Y.rows; ++row) {
    for (int col = 0; col < Y.cols; ++col) {
      float diff = Y_hat(row, col) - Y(row, col);
      grad(row, col) = scale * diff;
    }
  }

  return grad;
}

matrix relu(const matrix &a) {
  matrix out(a.rows, a.cols);

  for (int r = 0; r < a.rows; ++r) {
    for (int c = 0; c < a.cols; ++c) {
      float x = a(r, c);
      out(r, c) = x > 0.0f ? x : 0.0f;
    }
  }

  return out;
}

matrix relu_grad(const matrix &a) {
  matrix grad(a.rows, a.cols);

  for (int r = 0; r < a.rows; ++r) {
    for (int c = 0; c < a.cols; ++c) {
      grad(r, c) = a(r, c) > 0.0f ? 1.0f : 0.0f;
    }
  }

  return grad;
}

int main() {
  std::cout << "nn from scratch\n";

  matrix X(4, 1);
  X(0, 0) = 1;
  X(1, 0) = 2;
  X(2, 0) = 3;
  X(3, 0) = 4;

  matrix Y(4, 1);
  Y(0, 0) = 3;
  Y(1, 0) = 5;
  Y(2, 0) = 7;
  Y(3, 0) = 9;

  matrix W(1, 1);
  W(0, 0) = 0.1f;

  matrix b(4, 1);
  for (int i = 0; i < 4; ++i) {
    b(i, 0) = 0.0f;
  }

  float lr = 0.01f;

  for (int epoch = 0; epoch < 1000; ++epoch) {

    matrix Z = affine(X, W, b);
    matrix Y_hat = relu(Z);

    float loss = mean_squared_error(Y, Y_hat);

    matrix dA = mse_grad(Y, Y_hat);
    matrix dZ = hadamard(dA, relu_grad(Z));

    matrix X_T = transpose(X);
    matrix dW = matmul(X_T, dZ);

    // bias gradient
    matrix db(4, 1);
    for (int i = 0; i < 4; ++i)
      db(i, 0) = dZ(i, 0);

    W = mat_sub(W, scalar_mul(dW, lr));
    b = mat_sub(b, scalar_mul(db, lr));

    if (epoch % 100 == 0)
      std::cout << "loss: " << loss << "\n";
  }

  return 0;
}
