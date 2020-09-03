// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <pybind11/pybind11.h>

namespace rela {

class ModelLocker {
 public:
  ModelLocker(py::object pyModel, const std::string& deviceName, int numCopies)
      : device(torch::Device(deviceName))
      , modelCallCounts_(numCopies, 0)
      , latestModel_(0) {
    for (int i = 0; i < numCopies; ++i) {
      pyModels_.push_back(pyModel.attr("clone")(deviceName));
      models_.push_back(pyModels_[i].attr("_c").cast<torch::jit::script::Module*>());
    }
  }

  ModelLocker(py::object pyModel, const std::string& deviceName)
      : device(torch::Device(deviceName))
      , modelCallCounts_(1, 0)
      , latestModel_(0) {
    pyModels_.push_back(pyModel);
    models_.push_back(pyModels_[0].attr("_c").cast<torch::jit::script::Module*>());
  }

  void updateModel(py::object pyModel) {
    std::unique_lock<std::mutex> lk(m_);
    int id = (latestModel_ + 1) % modelCallCounts_.size();
    cv_.wait(lk, [this, id] { return modelCallCounts_[id] == 0; });
    lk.unlock();

    pyModels_[id].attr("load_state_dict")(pyModel.attr("state_dict")());

    lk.lock();
    latestModel_ = id;
    lk.unlock();
  }

  const torch::jit::script::Module& getModel(int* id) {
    std::lock_guard<std::mutex> lk(m_);
    *id = latestModel_;
    // std::cout << "using mdoel: " << latestModel_ << std::endl;
    ++modelCallCounts_[latestModel_];
    return *models_[latestModel_];
  }

  void releaseModel(int id) {
    std::unique_lock<std::mutex> lk(m_);
    --modelCallCounts_[id];
    if (modelCallCounts_[id] == 0) {
      cv_.notify_one();
    }
  }

  const torch::Device device;

 private:
  // py::function model_cons_;
  std::vector<py::object> pyModels_;
  std::vector<int> modelCallCounts_;
  int latestModel_;

  std::vector<torch::jit::script::Module*> models_;
  std::mutex m_;
  std::condition_variable cv_;
};

}  // namespace rela
