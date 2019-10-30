// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <atomic>

#include "rela/actor.h"
#include "rela/env.h"

namespace rela {

class ThreadLoop {
 public:
  ThreadLoop() = default;

  ThreadLoop(const ThreadLoop&) = delete;
  ThreadLoop& operator=(const ThreadLoop&) = delete;

  virtual ~ThreadLoop() {
  }

  virtual void terminate() {
    terminated_ = true;
  }

  virtual void pause() {
    std::lock_guard<std::mutex> lk(mPaused_);
    paused_ = true;
  }

  virtual void resume() {
    {
      std::lock_guard<std::mutex> lk(mPaused_);
      paused_ = false;
    }
    cvPaused_.notify_one();
  }

  virtual void waitUntilResume() {
    std::unique_lock<std::mutex> lk(mPaused_);
    cvPaused_.wait(lk, [this] { return !paused_; });
  }

  virtual bool terminated() {
    return terminated_;
  }

  virtual bool paused() {
    return paused_;
  }

  virtual void mainLoop() = 0;

 private:
  std::atomic_bool terminated_{false};

  std::mutex mPaused_;
  bool paused_ = false;
  std::condition_variable cvPaused_;
};

// a simple implementation of ThreadLoop for single agent env
class BasicThreadLoop : public ThreadLoop {
 public:
  BasicThreadLoop(std::shared_ptr<Actor> actor,
                  std::shared_ptr<VectorEnv> env,
                  bool eval)
      : actor_(std::move(actor))
      , env_(std::move(env))
      , eval_(eval) {
    if (eval_) {
      assert(env_->size() == 1);
    }
  }

  virtual void mainLoop() final {
    TensorDict obs = {};
    torch::Tensor r;
    torch::Tensor t;
    while (!terminated()) {
      obs = env_->reset(obs);
      while (!env_->anyTerminated()) {
        if (terminated()) {
          break;
        }

        if (paused()) {
          waitUntilResume();
        }

        auto action = actor_->act(obs);
        std::tie(obs, r, t) = env_->step(action);

        if (eval_) {
          continue;
        }

        actor_->setRewardAndTerminal(r, t);
        actor_->postStep();
      }

      // eval only runs for one game
      if (eval_) {
        break;
      }
    }
  }

 private:
  std::shared_ptr<Actor> actor_;
  std::shared_ptr<VectorEnv> env_;
  const bool eval_;
};
}  // namespace rela
