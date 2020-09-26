// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/utils.h"

namespace rela {

TensorDict allocateBatchStorage(const TensorDict& data, int size) {
  TensorDict storage;
  for (const auto& kv : data) {
    auto t = kv.second.sizes();
    std::vector<int64_t> sizes;
    // for (int i = 0; i < batchdim_; ++i) {
    //   sizes.push_back(t[i]);
    // }
    sizes.push_back(size);
    for (size_t i = 0; i < t.size(); ++i) {
      sizes.push_back(t[i]);
    }

    storage[kv.first] = torch::zeros(sizes, kv.second.dtype());
  }
  return storage;
}

class FutureReply {
 public:
  FutureReply()
      : ready_(false) {
  }

  TensorDict get(int slot) {
    // std::cout << "getting slot: " << slot << std::endl;
    std::unique_lock<std::mutex> lk(mReady_);
    cvReady_.wait(lk, [this] { return ready_; });
    lk.unlock();

    TensorDict e;
    for (const auto& kv : data_) {
      assert(slot >= 0 && slot < kv.second.size(0));
      e[kv.first] = kv.second[slot];
      // std::cout << kv.first << "\n" << e[kv.first] << std::endl;
    }
    return e;
    // return data_[slot];
  }

  void set(TensorDict&& t) {
    // assert(t.device().is_cpu());
    {
      std::lock_guard<std::mutex> lk(mReady_);
      ready_ = true;
      data_ = std::move(t);
    }
    cvReady_.notify_all();
  }

 private:
  // no need for protection, only set() can set it
  TensorDict data_;

  std::mutex mReady_;
  bool ready_;
  std::condition_variable cvReady_;
};

class Batcher {
 public:
  Batcher(int batchsize)
      : batchsize_(batchsize)
      , nextSlot_(0)
      , numActiveWrite_(0)
      , fillingReply_(std::make_shared<FutureReply>())
      , filledReply_(nullptr) {
  }

  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

  ~Batcher() {
    if (!exit_) {
      exit();
    }
  }

  void exit() {
    {
      std::unique_lock<std::mutex> lk(mNextSlot_);
      exit_ = true;
    }
    cvGetBatch_.notify_all();
  }

  void reset() {
    assert(exit_ == true);
    exit_ = false;
  }

  bool terminated() {
    return exit_;
  }

  // send data into batcher
  std::shared_ptr<FutureReply> send(const TensorDict& t, int* slot) {
    std::unique_lock<std::mutex> lk(mNextSlot_);

    // init buffer
    if (fillingBuffer_.empty()) {
      assert(filledBuffer_.empty());
      fillingBuffer_ = allocateBatchStorage(t, batchsize_);
      filledBuffer_ = allocateBatchStorage(t, batchsize_);
    }

    assert(nextSlot_ <= batchsize_);
    // wait if current batch is full and not extracted
    cvNextSlot_.wait(lk, [this] { return nextSlot_ < batchsize_; });

    *slot = nextSlot_;
    ++nextSlot_;
    ++numActiveWrite_;
    lk.unlock();

    // this will copy
    for (const auto& kv : t) {
      fillingBuffer_[kv.first][*slot] = kv.second;
    }

    // batch has not been extracted yet
    assert(numActiveWrite_ > 0);
    assert(fillingReply_ != nullptr);
    auto reply = fillingReply_;
    lk.lock();
    --numActiveWrite_;
    lk.unlock();
    if (numActiveWrite_ == 0) {
      cvGetBatch_.notify_one();
    }
    return reply;
  }

  // get batch input from batcher
  TensorDict get() {
    std::unique_lock<std::mutex> lk(mNextSlot_);
    cvGetBatch_.wait(lk, [this] {
      return (nextSlot_ > 0 && numActiveWrite_ == 0) || exit_;
    });

    if (exit_) {
      return TensorDict();
    }

    // TensorDict batch;
    // for (const auto& kv : buffer_) {
    //   batch[kv.first] = kv.second.narrow_copy(batchdim_, 0, nextSlot_).contiguous();
    // }
    int bsize = nextSlot_;
    nextSlot_ = 0;
    // assert previous reply has been handled
    assert(filledReply_ == nullptr);
    std::swap(fillingBuffer_, filledBuffer_);
    std::swap(fillingReply_, filledReply_);
    fillingReply_ = std::make_shared<FutureReply>();

    // assert currentReply has been handled
    // assert(currentReply_ == nullptr);
    // currentreply_ = std::move(nextReply_);
    // nextReply_ = std::make_shared<FutureReply>(batchdim_);

    lk.unlock();
    cvNextSlot_.notify_all();

    TensorDict batch;
    for (const auto& kv : filledBuffer_) {
      batch[kv.first] = kv.second.narrow(0, 0, bsize).contiguous();
      // batch[kv.first] = kv.second.narrow_copy(0, 0, batchsize_).contiguous();
    }

    sumBatchsize_ += bsize;
    batchCount_ += 1;
    if (batchCount_ % 5000 == 0) {
      /*
      if (sumBatchsize_ / batchCount_ > 100) {
        std::cout << ">>>>>>>>>>>>>>>.batchcount: " << (int64_t)this << std::endl;
        std::cout << sumBatchsize_ / (float)batchCount_ << std::endl;
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>"<< std::endl;
      }
      */
      sumBatchsize_ = 0;
      batchCount_ = 0;
    }

    return batch;
  }

  // set batch reply for batcher
  void set(TensorDict&& t) {
    // auto start = high_resolution_clock::now();

    for (const auto& kv : t) {
      assert(kv.second.device().is_cpu());
    }
    // assert(currentReply_ != nullptr);
    filledReply_->set(std::move(t));
    filledReply_ = nullptr;
  }

 private:
  const int batchsize_;

  int sumBatchsize_ = 0;
  int batchCount_ = 0;

  int nextSlot_;
  int numActiveWrite_;
  std::condition_variable cvNextSlot_;

  TensorDict fillingBuffer_;
  std::shared_ptr<FutureReply> fillingReply_;

  TensorDict filledBuffer_;
  std::shared_ptr<FutureReply> filledReply_;

  bool exit_ = false;
  std::condition_variable cvGetBatch_;
  std::mutex mNextSlot_;
  // std::map<std::string, float> timer_;
  // int counter_ = 0;
};

}  // namespace rela
