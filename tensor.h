#pragma once

#include <numeric>
#include <vector>
#include <cassert>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> stride;

    Tensor(std::vector<int> s) : shape(s) {
        int d = shape.size();
        assert(d > 0);
        stride.resize(d);

        stride[d-1] = 1;
        for (int i = d-2; i >= 0; --i) { 
            stride[i] = stride[i+1] * shape[i+1];
        }

        data.resize(numel());
    };

    Tensor(std::initializer_list<int> s) : Tensor(std::vector<int>(s)) {};

    int numel() const { return std::accumulate(
            shape.begin(),
            shape.end(),
            1,
            std::multiplies<int>()
            ); 
    };

    float& operator[](int i) { return data[i]; };
    float operator[](int i) const { return data[i]; };

    float& operator()(const std::vector<int>& idx) { 
        assert(idx.size() == shape.size());
        return data[offset(idx)];
    }
    float operator()(const std::vector<int>& idx) const { 
        assert(idx.size() == shape.size());
        return data[offset(idx)];
    }


    float& at(const std::vector<int>& idx) {
        assert(idx.size() == shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            assert(idx[i] >= 0);
            assert(idx[i] < shape[i]);
        }
        return data[offset(idx)];
    }
    float at(const std::vector<int>& idx) const {
        assert(idx.size() == shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            assert(idx[i] >= 0);
            assert(idx[i] < shape[i]);
        }
        return data[offset(idx)];
    }


    private:
        int offset(const std::vector<int>& idx) const {
            int off = 0;
            for (int i = 0; i < shape.size(); ++i)
                off += idx[i] * stride[i];
            return off;
        }
};
