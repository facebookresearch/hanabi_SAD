// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <thread>
#include <cassert>

#include "rela/tensor_dict.h"
#include "rela/batcher.h"

namespace rela {

class BatchRunner {
 public:
  BatchRunner(py::object pyModel,
              const std::string& device,
              int maxBatchsize,
              const std::vector<std::string>& methods)
      : pyModel_(pyModel)
      , jitModel_(pyModel_.attr("_c").cast<torch::jit::script::Module*>())
      , device_(torch::Device(device))
      , batchsize_(maxBatchsize)
      , methods_(methods) {
  }

  BatchRunner(const BatchRunner&) = delete;
  BatchRunner& operator=(const BatchRunner&) = delete;

  ~BatchRunner() {
    stop();
  }

  std::shared_ptr<FutureReply> call(const std::string& method, const TensorDict& t, int* slot) {
    auto batcherIt = batchers_.find(method);
    if (batcherIt == batchers_.end()) {
      std::cerr << "Error: Cannot find method: " << method << std::endl;
      assert(false);
    }
    return batcherIt->second->send(t, slot);
  }

  void start() {
    if (batchers_.empty()) {
      for (auto& name : methods_) {
        batchers_.emplace(name, std::make_unique<Batcher>(batchsize_));
      }
    } else {
      for (auto& kv : batchers_) {
        kv.second->reset();
      }
    }

    for (auto& kv : batchers_) {
      threads_.emplace_back(&BatchRunner::runnerLoop, this, kv.first);
    }
  }

  void stop() {
    for (auto& kv : batchers_) {
      kv.second->exit();
    }
    // batchers_.clear();

    for (auto& v : threads_) {
      v.join();
    }
    threads_.clear();
  }

  void updateModel(py::object agent) {
    std::lock_guard<std::mutex> lk(mtxUpdate_);
    pyModel_.attr("load_state_dict")(agent.attr("state_dict")());
  }

  const torch::jit::script::Module& jitModel() {
    return *jitModel_;
  }

 private:
  void runnerLoop(const std::string& method) {
    auto batcherIt = batchers_.find(method);
    if (batcherIt == batchers_.end()) {
      std::cerr << "Error: RunnerLoop, Cannot find method: " << method << std::endl;
      assert(false);
    }
    auto& batcher = *(batcherIt->second);

    while(!batcher.terminated()) {
      auto batch = batcher.get();
      if (batch.empty()) {
        assert(batcher.terminated());
        break;
      }

      {
        std::lock_guard<std::mutex> lk(mtxDevice_);

        torch::NoGradGuard ng;
        std::vector<torch::jit::IValue> input;
        input.push_back(tensor_dict::toIValue(batch, device_));
        torch::jit::IValue output;
        {
          std::lock_guard<std::mutex> lk(mtxUpdate_);
          output = jitModel_->get_method(method)(input);
        }
        batcher.set(tensor_dict::fromIValue(output, torch::kCPU, true));
      }
    }
  }

  py::object pyModel_;
  torch::jit::script::Module* const jitModel_;
  const torch::Device device_;
  const int batchsize_;
  const std::vector<std::string> methods_;

  // ideally this mutex should be 1 per device, thus global
  std::mutex mtxDevice_;
  std::mutex mtxUpdate_;
  bool updateDue_;
  py::object stateDict_;

  // std::unordered_map<std::string, Batcher> batchers_;
  std::map<std::string, std::unique_ptr<Batcher>> batchers_;
  std::vector<std::thread> threads_;
};
}
