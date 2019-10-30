// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <math.h>
#include <torch/script.h>

#include "rela/actor.h"
#include "rela/model_locker.h"
#include "rela/prioritized_replay.h"
#include "rela/utils.h"

namespace rela {

class MultiStepTransitionBuffer {
 public:
  MultiStepTransitionBuffer(int multiStep, int batchsize, float gamma)
      : multiStep_(multiStep)
      , batchsize_(batchsize)
      , gamma_(gamma) {
  }

  void pushObsAndAction(TensorDict& obs, TensorDict& action) {
    assert((int)obsHistory_.size() <= multiStep_);
    assert((int)actionHistory_.size() <= multiStep_);

    obsHistory_.push_back(obs);
    actionHistory_.push_back(action);
  }

  void pushRewardAndTerminal(torch::Tensor& reward, torch::Tensor& terminal) {
    assert(rewardHistory_.size() == terminalHistory_.size());
    assert(rewardHistory_.size() == obsHistory_.size() - 1);
    assert(reward.dim() == 1 && terminal.dim() == 1);
    assert(reward.size(0) == terminal.size(0) && reward.size(0) == batchsize_);

    rewardHistory_.push_back(reward);
    terminalHistory_.push_back(terminal);
  }

  size_t size() {
    return obsHistory_.size();
  }

  bool canPop() {
    return (int)obsHistory_.size() == multiStep_ + 1;
  }

  /* assumes that:
   *  obsHistory contains s_t, s_t+1, ..., s_t+n
   *  actionHistory contains a_t, a_t+1, ..., a_t+n
   *  rewardHistory contains r_t, r_t+1, ..., r_t+n
   *  terminalHistory contains T_t, T_t+1, ..., T_t+n
   *
   * returns:
   *  obs, action, cumulative reward, terminal, "next"_obs
   */
  FFTransition popTransition() {
    assert((int)obsHistory_.size() == multiStep_ + 1);
    assert((int)actionHistory_.size() == multiStep_ + 1);
    assert((int)rewardHistory_.size() == multiStep_ + 1);
    assert((int)terminalHistory_.size() == multiStep_ + 1);

    TensorDict obs = obsHistory_.front();
    TensorDict action = actionHistory_.front();
    torch::Tensor terminal = terminalHistory_.front();
    torch::Tensor bootstrap = torch::ones(batchsize_, torch::kFloat32);
    auto bootstrapAccessor = bootstrap.accessor<float, 1>();

    std::vector<int> nextObsIndices(batchsize_);
    // calculate bootstrap and nextState indices
    for (int i = 0; i < batchsize_; i++) {
      for (int step = 0; step < multiStep_; step++) {
        // next state is step (shouldn't be used anyways)
        if (terminalHistory_[step][i].item<bool>()) {
          bootstrapAccessor[i] = 0.0;
          nextObsIndices[i] = step;
          break;
        }
      }
      // next state is step+n
      if (bootstrapAccessor[i] > 1e-6) {
        nextObsIndices[i] = multiStep_;
      }
    }

    // calculate discounted rewards
    torch::Tensor reward = torch::zeros_like(rewardHistory_.front());
    auto accessor = reward.accessor<float, 1>();
    for (int i = 0; i < batchsize_; i++) {
      // if bootstrap, we only use the first nsAccessor[i]-1 (i.e. multiStep_-1)
      int initial = bootstrapAccessor[i] ? multiStep_ - 1 : nextObsIndices[i];
      for (int step = initial; step >= 0; step--) {
        float stepReward = rewardHistory_[step][i].item<float>();
        accessor[i] = stepReward + gamma_ * accessor[i];
      }
    }

    TensorDict nextObs = obsHistory_.back();

    obsHistory_.pop_front();
    actionHistory_.pop_front();
    rewardHistory_.pop_front();
    terminalHistory_.pop_front();
    return FFTransition(obs, action, reward, terminal, bootstrap, nextObs);
  }

  void clear() {
    obsHistory_.clear();
    actionHistory_.clear();
    rewardHistory_.clear();
    terminalHistory_.clear();
  }

 private:
  const int multiStep_;
  const int batchsize_;
  const float gamma_;

  std::deque<TensorDict> obsHistory_;
  std::deque<TensorDict> actionHistory_;
  std::deque<torch::Tensor> rewardHistory_;
  std::deque<torch::Tensor> terminalHistory_;
};

class DQNActor : public Actor {
 public:
  DQNActor(std::shared_ptr<ModelLocker> modelLocker,
           int multiStep,
           int batchsize,
           float gamma,
           std::shared_ptr<FFPrioritizedReplay> replayBuffer)
      : batchsize_(batchsize)
      , modelLocker_(std::move(modelLocker))
      , transitionBuffer_(multiStep, batchsize, gamma)
      , replayBuffer_(replayBuffer)
      , numAct_(0) {
  }

  // for single env evaluation
  DQNActor(std::shared_ptr<ModelLocker> modelLocker)
      : batchsize_(1)
      , modelLocker_(std::move(modelLocker))
      , transitionBuffer_(1, 1, 1)
      , replayBuffer_(nullptr)
      , numAct_(0) {
  }

  int numAct() const {
    return numAct_;
  }

  virtual TensorDict act(TensorDict& obs) override {
    torch::NoGradGuard ng;
    auto inputObs = utils::tensorDictToTorchDict(obs, modelLocker_->device);
    TorchJitInput input;
    input.push_back(inputObs);

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    TorchJitOutput output = model.get_method("act")(input);
    modelLocker_->releaseModel(id);

    auto action = utils::iValueToTensorDict(output, torch::kCPU, true);
    if (replayBuffer_ != nullptr) {
      transitionBuffer_.pushObsAndAction(obs, action);
    }

    numAct_ += batchsize_;
    return action;
  }

  // r is float32 tensor, t is bool tensor
  virtual void setRewardAndTerminal(torch::Tensor& r, torch::Tensor& t) override {
    assert(replayBuffer_ != nullptr);
    transitionBuffer_.pushRewardAndTerminal(r, t);
  }

  // should be called after setRewardAndTerminal
  // Pops a batch of transitions and inserts it into the replay buffer
  virtual void postStep() override {
    assert(replayBuffer_ != nullptr);
    if (!transitionBuffer_.canPop()) {
      return;
    }

    auto transition = transitionBuffer_.popTransition();
    auto priority = computePriority(transition);
    replayBuffer_->add(transition, priority);
  }

 private:
  torch::Tensor computePriority(const FFTransition& sample) {
    torch::NoGradGuard ng;
    int id = -1;
    auto input = sample.toJitInput(modelLocker_->device);

    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("compute_priority")(input);
    modelLocker_->releaseModel(id);

    return output.toTensor().detach().to(torch::kCPU);
  }

  const int batchsize_;

  std::shared_ptr<ModelLocker> modelLocker_;
  MultiStepTransitionBuffer transitionBuffer_;
  std::shared_ptr<FFPrioritizedReplay> replayBuffer_;
  std::atomic<int> numAct_;
};
}  // namespace rela
