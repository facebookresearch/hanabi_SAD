// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/actor.h"
#include "rela/dqn_actor.h"

namespace rela {

class R2D2TransitionBuffer {
 public:
  R2D2TransitionBuffer(
      int batchsize, int numPlayer, int multiStep, int seqLen)
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

  void push(const FFTransition& transition,
            const torch::Tensor& priority,
            const TensorDict&) {
    assert(priority.size(0) == batchsize);

    auto priorityAccessor = priority.accessor<float, 1>();
    for (int i = 0; i < batchsize; ++i) {
      int nextIdx = batchNextIdx_[i];
      assert(nextIdx < seqLen && nextIdx >= 0);
      if (nextIdx == 0) {
        // NOTE: simplification for unconditional h0
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

  std::tuple<std::vector<RNNTransition>, torch::Tensor, torch::Tensor>
  popTransition() {
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
      auto t = RNNTransition(batchSeqTransition_[i],
                             batchH0_[i],
                             torch::tensor(float(batchLen_[i])));
      batchTransition.push_back(t);

      batchLen_[i] = 0;
      batchNextIdx_[i] = 0;
    }
    canPop_ = false;
    assert(batchTransition.size() > 0);

    return std::make_tuple(batchTransition,
                           torch::stack(batchSeqPriority, 1), // batchdim = 1
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

class R2D2Actor : public Actor {
 public:
  R2D2Actor(std::shared_ptr<ModelLocker> modelLocker,
            int multiStep,
            int batchsize,
            float gamma,
            int seqLen,
            float greedyEps,
            int numPlayer,
            std::shared_ptr<RNNPrioritizedReplay> replayBuffer)
      : modelLocker_(std::move(modelLocker))
      , batchsize_(batchsize)
      , greedyEps_(greedyEps)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(batchsize, numPlayer, multiStep, seqLen)
      , multiStepBuffer_(multiStep, batchsize, gamma)
      , replayBuffer_(std::move(replayBuffer))
      , hidden_(getH0(batchsize * numPlayer))
      , numAct_(0) {
  }

  R2D2Actor(std::shared_ptr<ModelLocker> modelLocker,
            int numPlayer,
            float greedyEps)
      : modelLocker_(std::move(modelLocker))
      , batchsize_(1)
      , greedyEps_(greedyEps)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(1, numPlayer, 1, 1)  // never be used in this mode
      , multiStepBuffer_(1, 1, 1)        // never be used in this mode
      , replayBuffer_(nullptr)
      , hidden_(getH0(1 * numPlayer))
      , numAct_(0) {
  }

  int numAct() const {
    return numAct_;
  }

  virtual TensorDict act(TensorDict& obs) override {
    torch::NoGradGuard ng;
    assert(!hidden_.empty());

    if (replayBuffer_ != nullptr) {
      historyHidden_.push_back(hidden_);
    }

    auto eps = torch::zeros({batchsize_, numPlayer_}, torch::kFloat32);
    if (numPlayer_ == 1) {
      eps = eps.squeeze(1);
    }
    eps.fill_(greedyEps_);
    obs["eps"] = eps;

    TorchJitInput input;
    auto jitObs = utils::tensorDictToTorchDict(obs, modelLocker_->device);
    auto jitHid = utils::tensorDictToTorchDict(hidden_, modelLocker_->device);
    input.push_back(jitObs);
    input.push_back(jitHid);

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("act")(input).toTuple()->elements();
    modelLocker_->releaseModel(id);

    auto action = utils::iValueToTensorDict(output[0], torch::kCPU, true);
    hidden_ = utils::iValueToTensorDict(output[1], torch::kCPU, true);

    if (replayBuffer_ != nullptr) {
      multiStepBuffer_.pushObsAndAction(obs, action);
    }

    numAct_ += batchsize_;
    return action;
  }

  // r is float32 tensor, t is byte tensor
  virtual void setRewardAndTerminal(torch::Tensor& r,
                                    torch::Tensor& t) override {
    if (replayBuffer_ == nullptr) {
      return;
    }
    // assert(replayBuffer_ != nullptr);
    multiStepBuffer_.pushRewardAndTerminal(r, t);

    // if ith state is terminal, reset hidden states
    // h0: [num_layers * num_directions, batch, hidden_size]
    TensorDict h0 = getH0(1);
    auto terminal = t.accessor<bool, 1>();
    // std::cout << "terminal size: " << t.sizes() << std::endl;
    // std::cout << "hid size: " << hidden_["h0"].sizes() << std::endl;
    for (int i = 0; i < terminal.size(0); i++) {
      if (terminal[i]) {
        for (auto& name2tensor : hidden_) {
          // batch dim is 1
          name2tensor.second.narrow(1, i * numPlayer_, numPlayer_) =
              h0.at(name2tensor.first);
        }
      }
    }
  }

  // should be called after setRewardAndTerminal
  // Pops a batch of transitions and inserts it into the replay buffer
  virtual void postStep() override {
    if (replayBuffer_ == nullptr) {
      return;
    }
    // assert(replayBuffer_ != nullptr);
    assert(multiStepBuffer_.size() == historyHidden_.size());

    if (!multiStepBuffer_.canPop()) {
      assert(!r2d2Buffer_.canPop());
      return;
    }

    {
      FFTransition transition = multiStepBuffer_.popTransition();
      TensorDict hid = historyHidden_.front();
      TensorDict nextHid = historyHidden_.back();
      historyHidden_.pop_front();

      torch::Tensor priority = computePriority(transition, hid, nextHid);
      r2d2Buffer_.push(transition, priority, hid);
    }

    if (!r2d2Buffer_.canPop()) {
      return;
    }

    std::vector<RNNTransition> batch;
    torch::Tensor seqBatchPriority;
    torch::Tensor batchLen;

    std::tie(batch, seqBatchPriority, batchLen) = r2d2Buffer_.popTransition();
    auto priority = aggregatePriority(seqBatchPriority, batchLen);
    replayBuffer_->add(batch, priority);
  }

 private:
  TensorDict getH0(int batchsize) {
    TorchJitInput input;
    input.push_back(batchsize);
    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("get_h0")(input);
    modelLocker_->releaseModel(id);
    return utils::iValueToTensorDict(output, torch::kCPU, true);
  }

  torch::Tensor computePriority(const FFTransition& transition,
                                TensorDict hid,
                                TensorDict nextHid) {
    torch::NoGradGuard ng;
    auto device = modelLocker_->device;
    auto input = transition.toJitInput(device);
    input.push_back(utils::tensorDictToTorchDict(hid, device));
    input.push_back(utils::tensorDictToTorchDict(nextHid, device));

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto priority = model.get_method("compute_priority")(input).toTensor();
    modelLocker_->releaseModel(id);
    return priority.detach().to(torch::kCPU);
  }

  torch::Tensor aggregatePriority(torch::Tensor priority, torch::Tensor len) {
    // priority: [batchsize, seqLen]
    TorchJitInput input;
    input.push_back(priority);
    input.push_back(len);
    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto aggPriority = model.get_method("aggregate_priority")(input).toTensor();
    modelLocker_->releaseModel(id);
    return aggPriority;
  }

  std::shared_ptr<ModelLocker> modelLocker_;
  const int batchsize_;
  const float greedyEps_;
  const int numPlayer_;

  std::deque<TensorDict> historyHidden_;
  R2D2TransitionBuffer r2d2Buffer_;
  MultiStepTransitionBuffer multiStepBuffer_;
  std::shared_ptr<RNNPrioritizedReplay> replayBuffer_;

  TensorDict hidden_;
  std::atomic<int> numAct_;
};
}
