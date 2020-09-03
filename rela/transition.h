// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <torch/extension.h>

#include "tensor_dict.h"

namespace rela {

class FFTransition {
 public:
  FFTransition() = default;

  FFTransition(
      TensorDict& obs,
      TensorDict& action,
      torch::Tensor& reward,
      torch::Tensor& terminal,
      torch::Tensor& bootstrap,
      TensorDict& nextObs)
      : obs(obs)
      , action(action)
      , reward(reward)
      , terminal(terminal)
      , bootstrap(bootstrap)
      , nextObs(nextObs) {
  }

  FFTransition index(int i) const;

  FFTransition padLike() const;

  std::vector<torch::jit::IValue> toVectorIValue(const torch::Device& device) const;

  TensorDict toDict();

  static FFTransition makeBatch(
      const std::vector<FFTransition>& transitions, const std::string& device);

  TensorDict obs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  TensorDict nextObs;
};

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition(
      const std::vector<FFTransition>& transitions, TensorDict h0, torch::Tensor seqLen);

  RNNTransition index(int i) const;

  static RNNTransition makeBatch(
      const std::vector<RNNTransition>& transitions, const std::string& device);

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;
};

}  // namespace rela
