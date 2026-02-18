#include <cassert>
#include <iostream>

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
    for (int col = 0; col < a.cols; ++col) {
      float sum = 0.0f;

      for (int k = 0; k < a.cols; ++k) {
        sum += a(row, k) * b(k, col);
      }

      out(row, col) = sum;
    }
  }

  return out;
}

matrix add(const matrix &a, const matrix &b) {
  assert(a.rows == b.rows && a.cols == b.cols);

  matrix out(a.rows, a.cols);
  for (int row = 0; row < a.rows; ++row) {
    for (int col = 0; col < a.cols; ++col) {

      out(row, col) = a(row, col) + b(row, col);
    }
  }

  return out;
};

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
  n(1, 2) = 4;
  n(1, 0) = 3;
  n(2, 1) = 2;
  n(2, 2) = 1;

  matrix q(2,3);

  q(0, 0) = 3;
  q(0, 1) = 5;
  q(0, 2) = 7;
  q(1, 0) = 7;
  q(1, 1) = 10;
  q(1, 2) = 6;

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      std::cout << m(r, c) << " ";
    }
    std::cout << "\n";
  }

  matrix answer = matmul(m, n);

  matrix add_answer = add(m,q);

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
