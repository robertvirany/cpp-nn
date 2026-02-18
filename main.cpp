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

int main() {
  std::cout << "nn from scratch\n";

  matrix m(2, 3);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  matrix n(3, 2);
  n(0, 0) = 6;
  n(0, 1) = 5;
  n(1, 0) = 4;
  n(1, 1) = 3;
  n(2, 0) = 2;
  n(2, 1) = 1;

  matrix q(2, 3);

  q(0, 0) = 3;
  q(0, 1) = 5;
  q(0, 2) = 7;
  q(1, 0) = 7;
  q(1, 1) = 10;
  q(1, 2) = 6;

  for (int r = 0; r < m.rows; ++r) {
    for (int c = 0; c < m.cols; ++c) {
      std::cout << m(r, c) << " ";
    }
    std::cout << "\n";
  }

  for (int r = 0; r < n.rows; ++r) {
    for (int c = 0; c < n.cols; ++c) {
      std::cout << n(r, c) << " ";
    }
    std::cout << "\n";
  }

  matrix answer = matmul(m, n);

  matrix add_answer = matadd(m, q);

  for (int r = 0; r < answer.rows; ++r) {
    for (int c = 0; c < answer.cols; ++c) {
      std::cout << answer(r, c) << " ";
    }
    std::cout << "\n";
  }

  for (int r = 0; r < add_answer.rows; ++r) {
    for (int c = 0; c < add_answer.cols; ++c) {
      std::cout << add_answer(r, c) << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
