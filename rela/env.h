// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <future>
#include <vector>

#include "rela/types.h"
#include "rela/utils.h"

namespace rela {

class Env {
 public:
  Env() = default;

  virtual ~Env() {
  }

  virtual TensorDict reset() = 0;

  // return 'obs', 'reward', 'terminal'
  virtual std::tuple<TensorDict, float, bool> step(
      const TensorDict& action) = 0;

  virtual bool terminated() const = 0;
};

// a "container" as if it is a vector of envs
class VectorEnv {
 public:
  VectorEnv() = default;

  virtual ~VectorEnv() {
  }

  void append(std::shared_ptr<Env> env) {
    envs_.push_back(std::move(env));
  }

  int size() const {
    return envs_.size();
  }

  // reset envs that have reached end of terminal
  virtual TensorDict reset(const TensorDict& input) {
    TensorVecDict batch;
    for (size_t i = 0; i < envs_.size(); i++) {
      if (envs_[i]->terminated()) {
        TensorDict obs = envs_[i]->reset();
        utils::tensorVecDictAppend(batch, obs);
      } else {
        assert(!input.empty());
        utils::tensorVecDictAppend(batch, utils::tensorDictIndex(input, i));
      }
    }
    return utils::tensorDictJoin(batch, 0);
  }

  // return 'obs', 'reward', 'terminal'
  // obs: [num_envs, obs_dims]
  // reward: float32 tensor [num_envs]
  // terminal: bool tensor [num_envs]
  virtual std::tuple<TensorDict, torch::Tensor, torch::Tensor> step(
      const TensorDict& action) {
    TensorVecDict batchObs;
    torch::Tensor batchReward = torch::zeros(envs_.size(), torch::kFloat32);
    torch::Tensor batchTerminal = torch::zeros(envs_.size(), torch::kBool);
    for (size_t i = 0; i < envs_.size(); i++) {
      TensorDict obs;
      float reward;
      bool terminal;

      auto a = utils::tensorDictIndex(action, i);
      std::tie(obs, reward, terminal) = envs_[i]->step(a);

      utils::tensorVecDictAppend(batchObs, obs);
      batchReward[i] = reward;
      batchTerminal[i] = terminal;
    }
    return std::make_tuple(
        utils::tensorDictJoin(batchObs, 0), batchReward, batchTerminal);
  }

  virtual bool anyTerminated() const {
    for (size_t i = 0; i < envs_.size(); i++) {
      if (envs_[i]->terminated())
        return true;
    }
    return false;
  }

  virtual bool allTerminated() const {
    for (size_t i = 0; i < envs_.size(); i++) {
      if (!envs_[i]->terminated())
        return false;
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<Env>> envs_;
};
}  // namespace rela
