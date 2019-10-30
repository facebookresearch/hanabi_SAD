// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "rela/thread_loop.h"

namespace rela {

class Context {
 public:
  Context()
      : started_(false)
      , numTerminatedThread_(0) {
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ~Context() {
    for (auto& v : loops_) {
      v->terminate();
    }
    for (auto& v : threads_) {
      v.join();
    }
  }

  int pushThreadLoop(std::shared_ptr<ThreadLoop> env) {
    assert(!started_);
    loops_.push_back(std::move(env));
    return (int)loops_.size();
  }

  void start() {
    for (int i = 0; i < (int)loops_.size(); ++i) {
      threads_.emplace_back([this, i]() {
        loops_[i]->mainLoop();
        ++numTerminatedThread_;
      });
    }
  }

  void pause() {
    for (auto& v : loops_) {
      v->pause();
    }
  }

  void resume() {
    for (auto& v : loops_) {
      v->resume();
    }
  }

  void terminate() {
    for (auto& v : loops_) {
      v->terminate();
    }
  }

  bool terminated() {
    // std::cout << ">>> " << numTerminatedThread_ << std::endl;
    return numTerminatedThread_ == (int)loops_.size();
  }

 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<ThreadLoop>> loops_;
  std::vector<std::thread> threads_;
};
}  // namespace rela
