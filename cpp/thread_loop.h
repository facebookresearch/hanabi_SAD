// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/r2d2_actor.h"
#include "rela/thread_loop.h"

using HanabiVecEnv = rela::VectorEnv<HanabiEnv>;

class HanabiThreadLoop : public rela::ThreadLoop {
 public:
  HanabiThreadLoop(
      std::shared_ptr<rela::R2D2Actor> actor,
      std::shared_ptr<HanabiVecEnv> vecEnv,
      bool eval)
      : actors_({std::move(actor)})
      , vecEnv_(std::move(vecEnv))
      , eval_(eval) {
    assert(actors_.size() >= 1);
    if (eval_) {
      assert(vecEnv_->size() == 1);
    }
  }

  HanabiThreadLoop(
      std::vector<std::shared_ptr<rela::R2D2Actor>> actors,
      std::shared_ptr<HanabiVecEnv> vecEnv,
      bool eval)
      : actors_(std::move(actors))
      , vecEnv_(std::move(vecEnv))
      , eval_(eval) {
    assert(actors_.size() >= 1);
    if (eval_) {
      assert(vecEnv_->size() == 1);
    }
  }

  void mainLoop() final {
    rela::TensorDict obs = {};
    torch::Tensor r;
    torch::Tensor t;
    while (!terminated()) {
      obs = vecEnv_->reset(obs);
      while (!vecEnv_->anyTerminated()) {
        if (terminated()) {
          break;
        }

        if (paused()) {
          waitUntilResume();
        }

        rela::TensorDict reply;
        if (actors_.size() == 1) {
          reply = actors_[0]->act(obs);
        } else {
          std::vector<rela::TensorDict> replyVec;
          for (int i = 0; i < (int)actors_.size(); ++i) {
            auto input = rela::tensor_dict::narrow(obs, 1, i, 1, true);
            // if (!logFile_.empty()) {
            //   logState(*file, input);
            // }
            auto rep = actors_[i]->act(input);
            replyVec.push_back(rep);
          }
          reply = rela::tensor_dict::stack(replyVec, 1);
        }
        std::tie(obs, r, t) = vecEnv_->step(reply);

        if (eval_) {
          continue;
        }

        for (int i = 0; i < (int)actors_.size(); ++i) {
          actors_[i]->postAct(r, t);
        }
      }

      // eval only runs for one game
      if (eval_) {
        break;
      }
    }
  }

 private:
  std::vector<std::shared_ptr<rela::R2D2Actor>> actors_;
  std::shared_ptr<HanabiVecEnv> vecEnv_;
  const bool eval_;
};
