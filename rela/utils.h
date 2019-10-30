// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <iostream>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "rela/types.h"

namespace rela {

namespace utils {

inline int getProduct(const std::vector<int64_t>& nums) {
  int prod = 1;
  for (auto v : nums) {
    prod *= v;
  }
  return prod;
}

template <typename T>
inline std::vector<T> pushLeft(T left, const std::vector<T>& vals) {
  std::vector<T> vec;
  vec.reserve(1 + vals.size());
  vec.push_back(left);
  for (auto v : vals) {
    vec.push_back(v);
  }
  return vec;
}

template <typename T>
inline void printVector(const std::vector<T>& vec) {
  for (const auto& v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMapKey(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMap(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ": " << name2sth.second << std::endl;
  }
  // std::cout << std::endl;
}

inline void verifyTensors(const TensorDict& src, const TensorDict& dest) {
  if (src.size() != dest.size()) {
    std::cout << "src.size()[" << src.size() << "] != dest.size()[" << dest.size() << "]"
              << std::endl;
    std::cout << "src keys: ";
    for (const auto& p : src)
      std::cout << p.first << " ";
    std::cout << "dest keys: ";
    for (const auto& p : dest)
      std::cout << p.first << " ";
    std::cout << std::endl;
    assert(false);
  }

  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    // std::cout << "in copy: trying to get: " << name << std::endl;
    // std::cout << "dest map keys" << std::endl;
    // printMapKey(dest);
    const auto& destTensor = dest.at(name);
    // if (destTensor.sizes() != srcTensor.sizes()) {
    //   std::cout << "copy size-mismatch: "
    //             << destTensor.sizes() << ", " << srcTensor.sizes() <<
    //             std::endl;
    // }
    if (destTensor.sizes() != srcTensor.sizes()) {
      std::cout << name << ", dstSize: " << destTensor.sizes()
                << ", srcSize: " << srcTensor.sizes() << std::endl;
      assert(false);
    }

    if (destTensor.dtype() != srcTensor.dtype()) {
      std::cout << name << ", dstType: " << destTensor.dtype()
                << ", srcType: " << srcTensor.dtype() << std::endl;
      assert(false);
    }
  }
}

inline void copyTensors(const TensorDict& src, TensorDict& dest) {
  verifyTensors(src, dest);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    // std::cout << "in copy: trying to get: " << name << std::endl;
    // std::cout << "dest map keys" << std::endl;
    // printMapKey(dest);
    auto& destTensor = dest.at(name);
    // if (destTensor.sizes() != srcTensor.sizes()) {
    //   std::cout << "copy size-mismatch: "
    //             << destTensor.sizes() << ", " << srcTensor.sizes() <<
    //             std::endl;
    // }
    destTensor.copy_(srcTensor);
  }
}

// // TODO: maybe merge these two functions?
// inline void copyTensors(
//     const std::unordered_map<std::string, torch::Tensor>& src,
//     std::unordered_map<std::string, torch::Tensor>& dest,
//     std::vector<int64_t>& index) {
//   assert(src.size() == dest.size());
//   assert(!index.empty());
//   torch::Tensor indexTensor =
//       torch::from_blob(index.data(), {(int64_t)index.size()}, torch::kInt64);

//   for (const auto& name2tensor : src) {
//     const auto& name = name2tensor.first;
//     const auto& srcTensor = name2tensor.second;
//     auto& destTensor = dest.at(name);
//     // assert(destTensor.sizes() == srcTensor.sizes());
//     assert(destTensor.dtype() == srcTensor.dtype());
//     assert(indexTensor.size(0) == srcTensor.size(0));
//     destTensor.index_copy_(0, indexTensor, srcTensor);
//   }
// }

inline void copyTensors(const TensorDict& src,
                        TensorDict& dest,
                        const torch::Tensor& index) {
  assert(src.size() == dest.size());
  assert(index.size(0) > 0);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    assert(destTensor.dtype() == srcTensor.dtype());
    assert(index.size(0) == srcTensor.size(0));
    destTensor.index_copy_(0, index, srcTensor);
  }
}

inline bool tensorDictEq(const TensorDict& d0, const TensorDict& d1) {
  if (d0.size() != d1.size()) {
    return false;
  }

  for (const auto& name2tensor : d0) {
    auto key = name2tensor.first;
    if ((d1.at(key) != name2tensor.second).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

/*
 * indexes into a TensorDict
 */
inline TensorDict tensorDictIndex(const TensorDict& batch, size_t i) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second[i]});
  }
  return result;
}

inline TensorDict tensorDictNarrow(
    const TensorDict& batch, size_t dim, size_t i, size_t len, bool squeeze) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    auto t = name2tensor.second.narrow(dim, i, len);
    if (squeeze) {
      assert(len == 1);
      t = t.squeeze(dim);
    }
    result.insert({name2tensor.first, std::move(t)});
  }
  return result;
}

/*
 * Given a TensorVecDict, returns a Tensor that concats them together
 */
inline TensorDict tensorDictJoin(const TensorVecDict& batch, int stackdim) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    result.insert({name2tensor.first, torch::stack(name2tensor.second, stackdim)});
  }
  return result;
}

inline TensorDict tensorDictClone(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, name2tensor.second.clone()});
  }
  return output;
}

inline TensorDict tensorDictZerosLike(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, torch::zeros_like(name2tensor.second)});
  }
  return output;
}

// TODO: rewrite the above functions with this template
template <typename Func>
inline TensorDict tensorDictApply(TensorDict& dict, Func f) {
  TensorDict output;
  for (const auto& name2tensor : dict) {
    auto tensor = f(name2tensor.second);
    output.insert({name2tensor.first, tensor});
  }
  return output;
}

// TODO: remove this function if not used
/*
 * Appends a TensorDict to a TensorVecDict.
 * If batch is empty, initialize it as input.
 * This function should be used in conjunction with tensorDictJoin to batch
 * together many TensorDicts
 */
inline void tensorVecDictAppend(TensorVecDict& batch, const TensorDict& input) {
  for (auto& name2tensor : input) {
    auto it = batch.find(name2tensor.first);
    if (it == batch.end()) {
      std::vector<torch::Tensor> singleton = {name2tensor.second};
      batch.insert({name2tensor.first, singleton});
    } else {
      it->second.push_back(name2tensor.second);
    }
  }
}

inline TensorDict vectorTensorDictJoin(const std::vector<TensorDict>& vec, int stackdim) {
  assert(vec.size() >= 1);
  TensorVecDict ret;
  for (auto& name2tensor : vec[0]) {
    ret[name2tensor.first] = std::vector<torch::Tensor>();
  }
  for (int i = 0; i < (int)vec.size(); ++i) {
    for (auto& name2tensor : vec[i]) {
      ret[name2tensor.first].push_back(name2tensor.second);
    }
  }
  return tensorDictJoin(ret, stackdim);
}

inline TensorDict iValueToTensorDict(const torch::IValue& value,
                                     torch::DeviceType device,
                                     bool detach) {
  std::unordered_map<std::string, torch::Tensor> map;
  auto dict = value.toGenericDict();
  // auto ivalMap = dict->elements();
  for (auto& name2tensor : dict) {
    auto name = name2tensor.key().toString();
    torch::Tensor tensor = name2tensor.value().toTensor();
    tensor = tensor.to(device);
    if (detach) {
      tensor = tensor.detach();
    }
    map.insert({name->string(), tensor});
  }
  return map;
}

// TODO: this may be simplified with constructor in the future version
inline TorchTensorDict tensorDictToTorchDict(const TensorDict& tensorDict,
                                             const torch::Device& device) {
  TorchTensorDict dict;
  for (const auto& name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return dict;
}
}  // namespace utils
}  // namespace rela
