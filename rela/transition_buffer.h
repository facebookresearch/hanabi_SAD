// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once

#include "transition.h"

namespace rela {

class MultiStepBuffer {
 public:
  MultiStepBuffer(int multiStep, int batchsize, float gamma)
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
    auto reward = torch::zeros_like(rewardHistory_.front());
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

class R2D2Buffer {
 public:
  R2D2Buffer(int batchsize, int numPlayer, int multiStep, int seqLen)
      : batchsize(batchsize)
      , numPlayer(numPlayer)
      , multiStep(multiStep)
      , seqLen(seqLen)
      , batchNextIdx_(batchsize, 0)
      , batchH0_(batchsize)
      , batchSeqTransition_(batchsize, std::vector<FFTransition>(seqLen))
      , batchSeqPriority_(batchsize, std::vector<float>(seqLen))
      , batchLen_(batchsize, 0)
      , canPop_(false) {
  }

  void push(
      const FFTransition& transition,
      const torch::Tensor& priority,
      const TensorDict& /*hid*/) {
    assert(priority.size(0) == batchsize);

    auto priorityAccessor = priority.accessor<float, 1>();
    for (int i = 0; i < batchsize; ++i) {
      int nextIdx = batchNextIdx_[i];
      assert(nextIdx < seqLen && nextIdx >= 0);
      if (nextIdx == 0) {
        // TODO: !!! simplification for unconditional h0
        // batchH0_[i] =
        //     utils::tensorDictNarrow(hid, 1, i * numPlayer, numPlayer, false);
      }

      auto t = transition.index(i);
      // some sanity check for termination
      if (nextIdx != 0) {
        // should not append after terminal
        // terminal should be processed when it is pushed
        assert(!batchSeqTransition_[i][nextIdx - 1].terminal.item<bool>());
        assert(batchLen_[i] == 0);
      }

      batchSeqTransition_[i][nextIdx] = t;
      batchSeqPriority_[i][nextIdx] = priorityAccessor[i];

      ++batchNextIdx_[i];
      if (!t.terminal.item<bool>()) {
        continue;
      }

      // pad the rest of the seq in case of terminal
      batchLen_[i] = batchNextIdx_[i];
      while (batchNextIdx_[i] < seqLen) {
        batchSeqTransition_[i][batchNextIdx_[i]] = t.padLike();
        batchSeqPriority_[i][batchNextIdx_[i]] = 0;
        ++batchNextIdx_[i];
      }
      canPop_ = true;
    }
  }

  bool canPop() {
    return canPop_;
  }

  std::tuple<std::vector<RNNTransition>, torch::Tensor, torch::Tensor> popTransition() {
    assert(canPop_);

    std::vector<RNNTransition> batchTransition;
    std::vector<torch::Tensor> batchSeqPriority;
    std::vector<float> batchLen;

    for (int i = 0; i < batchsize; ++i) {
      if (batchLen_[i] == 0) {
        continue;
      }
      assert(batchNextIdx_[i] == seqLen);

      batchSeqPriority.push_back(torch::tensor(batchSeqPriority_[i]));
      batchLen.push_back((float)batchLen_[i]);
      auto t = RNNTransition(
          batchSeqTransition_[i], batchH0_[i], torch::tensor(float(batchLen_[i])));
      batchTransition.push_back(t);

      batchLen_[i] = 0;
      batchNextIdx_[i] = 0;
    }
    canPop_ = false;
    assert(batchTransition.size() > 0);

    return std::make_tuple(
        batchTransition,
        torch::stack(batchSeqPriority, 1),  // batchdim = 1
        torch::tensor(batchLen));
  }

  const int batchsize;
  const int numPlayer;
  const int multiStep;
  const int seqLen;

 private:
  std::vector<int> batchNextIdx_;
  std::vector<TensorDict> batchH0_;

  std::vector<std::vector<FFTransition>> batchSeqTransition_;
  std::vector<std::vector<float>> batchSeqPriority_;
  std::vector<int> batchLen_;

  bool canPop_;
};
}
