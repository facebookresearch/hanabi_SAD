// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <future>
#include <vector>

#include "rela/tensor_dict.h"
#include "rela/utils.h"

namespace rela {

class Env {
 public:
  Env() = default;

  virtual ~Env() {
  }

  virtual TensorDict reset() = 0;

  // return 'obs', 'reward', 'terminal'
  virtual std::tuple<TensorDict, float, bool> step(const TensorDict& action) = 0;

  virtual bool terminated() const = 0;
};

// a "container" as if it is a vector of envs
template <
    typename EnvType,
    typename = std::enable_if_t<std::is_base_of<Env, EnvType>::value>>
class VectorEnv {
 public:
  VectorEnv() = default;

  virtual ~VectorEnv() {
  }

  void append(std::shared_ptr<EnvType> env) {
    envs_.push_back(std::move(env));
  }

  int size() const {
    return envs_.size();
  }

  // reset envs that have reached end of terminal
  virtual TensorDict reset(const TensorDict& input) {
    std::vector<TensorDict> batch;
    for (size_t i = 0; i < envs_.size(); i++) {
      if (envs_[i]->terminated()) {
        TensorDict obs = envs_[i]->reset();
        batch.push_back(obs);
      } else {
        assert(!input.empty());
        batch.push_back(tensor_dict::index(input, i));
      }
    }
    return tensor_dict::stack(batch, 0);
  }

  // return 'obs', 'reward', 'terminal'
  // obs: [num_envs, obs_dims]
  // reward: float32 tensor [num_envs]
  // terminal: bool tensor [num_envs]
  virtual std::tuple<TensorDict, torch::Tensor, torch::Tensor> step(
      const TensorDict& action) {
    std::vector<TensorDict> vObs;
    std::vector<float> vReward(envs_.size());
    std::vector<float> vTerminal(envs_.size());
    for (size_t i = 0; i < envs_.size(); i++) {
      TensorDict obs;
      float reward;
      bool terminal;

      auto a = tensor_dict::index(action, i);
      std::tie(obs, reward, terminal) = envs_[i]->step(a);

      vObs.push_back(obs);
      vReward[i] = reward;
      vTerminal[i] = (float)terminal;
    }
    auto batchObs = tensor_dict::stack(vObs, 0);
    auto batchReward = torch::tensor(vReward);
    auto batchTerminal = torch::tensor(vTerminal).to(torch::kBool);
    return std::make_tuple(batchObs, batchReward, batchTerminal);
  }

  virtual bool anyTerminated() const {
    for (size_t i = 0; i < envs_.size(); i++) {
      if (envs_[i]->terminated()) {
        return true;
      }
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
  std::vector<std::shared_ptr<EnvType>> envs_;
};
}  // namespace rela
