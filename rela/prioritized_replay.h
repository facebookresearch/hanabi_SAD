// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <future>
#include <random>
#include <torch/extension.h>
#include <vector>

#include "rela/tensor_dict.h"
#include "rela/transition.h"

namespace rela {

template <class DataType>
class ConcurrentQueue {
 public:
  ConcurrentQueue(int capacity)
      : capacity(capacity)
      , head_(0)
      , tail_(0)
      , size_(0)
      , safeTail_(0)
      , safeSize_(0)
      , sum_(0)
      , evicted_(capacity, false)
      , elements_(capacity)
      , weights_(capacity, 0) {
  }

  int safeSize(float* sum) const {
    std::unique_lock<std::mutex> lk(m_);
    if (sum != nullptr) {
      *sum = sum_;
    }
    return safeSize_;
  }

  int size() const {
    std::unique_lock<std::mutex> lk(m_);
    return size_;
  }

  void blockAppend(const std::vector<DataType>& block, const torch::Tensor& weights) {
    int blockSize = block.size();

    std::unique_lock<std::mutex> lk(m_);
    cvSize_.wait(lk, [=] { return size_ + blockSize <= capacity; });

    int start = tail_;
    int end = (tail_ + blockSize) % capacity;

    tail_ = end;
    size_ += blockSize;
    checkSize(head_, tail_, size_);

    lk.unlock();

    float sum = 0;
    auto weightAcc = weights.accessor<float, 1>();
    assert(weightAcc.size(0) == blockSize);
    for (int i = 0; i < blockSize; ++i) {
      int j = (start + i) % capacity;
      elements_[j] = block[i];
      weights_[j] = weightAcc[i];
      sum += weightAcc[i];
    }

    lk.lock();

    cvTail_.wait(lk, [=] { return safeTail_ == start; });
    safeTail_ = end;
    safeSize_ += blockSize;
    sum_ += sum;
    checkSize(head_, safeTail_, safeSize_);

    lk.unlock();
    cvTail_.notify_all();
  }

  // ------------------------------------------------------------- //
  // blockPop, update are thread-safe against blockAppend
  // but they are NOT thread-safe against each other

  void blockPop(int blockSize) {
    double diff = 0;
    int head = head_;
    for (int i = 0; i < blockSize; ++i) {
      diff -= weights_[head];
      evicted_[head] = true;
      head = (head + 1) % capacity;
    }

    {
      std::lock_guard<std::mutex> lk(m_);
      sum_ += diff;
      head_ = head;
      safeSize_ -= blockSize;
      size_ -= blockSize;
      assert(safeSize_ >= 0);
      checkSize(head_, safeTail_, safeSize_);
    }
    cvSize_.notify_all();
  }

  void update(const std::vector<int>& ids, const torch::Tensor& weights) {
    double diff = 0;
    auto weightAcc = weights.accessor<float, 1>();
    for (int i = 0; i < (int)ids.size(); ++i) {
      auto id = ids[i];
      if (evicted_[id]) {
        continue;
      }
      diff += (weightAcc[i] - weights_[id]);
      weights_[id] = weightAcc[i];
    }

    std::lock_guard<std::mutex> lk_(m_);
    sum_ += diff;
  }

  // ------------------------------------------------------------- //
  // accessing elements is never locked, operate safely!

  DataType get(int idx) {
    int id = (head_ + idx) % capacity;
    return elements_[id];
  }

  DataType getElementAndMark(int idx) {
    int id = (head_ + idx) % capacity;
    evicted_[id] = false;
    return elements_[id];
  }

  float getWeight(int idx, int* id) {
    assert(id != nullptr);
    *id = (head_ + idx) % capacity;
    return weights_[*id];
  }

  const int capacity;

 private:
  void checkSize(int head, int tail, int size) {
    if (size == 0) {
      assert(tail == head);
    } else if (tail > head) {
      if (tail - head != size) {
        std::cout << "tail-head: " << tail - head << " vs size: " << size << std::endl;
      }
      assert(tail - head == size);
    } else {
      if (tail + capacity - head != size) {
        std::cout << "tail-head: " << tail + capacity - head << " vs size: " << size
                  << std::endl;
      }
      assert(tail + capacity - head == size);
    }
  }

  mutable std::mutex m_;
  std::condition_variable cvSize_;
  std::condition_variable cvTail_;

  int head_;
  int tail_;
  int size_;

  int safeTail_;
  int safeSize_;
  double sum_;
  std::vector<bool> evicted_;

  std::vector<DataType> elements_;
  std::vector<float> weights_;
};

template <class DataType>
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta, int prefetch)
      : alpha_(alpha)  // priority exponent
      , beta_(beta)    // importance sampling exponent
      , prefetch_(prefetch)
      , capacity_(capacity)
      , storage_(int(1.25 * capacity))
      , numAdd_(0) {
    rng_.seed(seed);
  }

  void add(const std::vector<DataType>& sample, const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    auto weights = torch::pow(priority, alpha_);
    storage_.blockAppend(sample, weights);
    numAdd_ += priority.size(0);
  }

  void add(const DataType& sample, const torch::Tensor& priority) {
    std::vector<DataType> vec;
    int n = priority.size(0);
    for (int i = 0; i < n; ++i) {
      vec.push_back(sample.index(i));
    }
    add(vec, priority);
  }

  std::tuple<DataType, torch::Tensor> sample(int batchsize, const std::string& device) {
    if (!sampledIds_.empty()) {
      std::cout << "Error: previous samples' priority has not been updated." << std::endl;
      assert(false);
    }

    DataType batch;
    torch::Tensor priority;
    if (prefetch_ == 0) {
      std::tie(batch, priority, sampledIds_) = sample_(batchsize, device);
      return std::make_tuple(batch, priority);
    }

    if (futures_.empty()) {
      std::tie(batch, priority, sampledIds_) = sample_(batchsize, device);
    } else {
      // assert(futures_.size() == 1);
      std::tie(batch, priority, sampledIds_) = futures_.front().get();
      futures_.pop();
    }

    while ((int)futures_.size() < prefetch_) {
      auto f = std::async(
          std::launch::async,
          &PrioritizedReplay<DataType>::sample_,
          this,
          batchsize,
          device);
      futures_.push(std::move(f));
    }

    return std::make_tuple(batch, priority);
  }

  void updatePriority(const torch::Tensor& priority) {
    if (priority.size(0) == 0) {
      sampledIds_.clear();
      return;
    }

    assert(priority.dim() == 1);
    assert((int)sampledIds_.size() == priority.size(0));

    auto weights = torch::pow(priority, alpha_);
    {
      std::lock_guard<std::mutex> lk(mSampler_);
      storage_.update(sampledIds_, weights);
    }
    sampledIds_.clear();
  }

  DataType get(int idx) {
    return storage_.get(idx);
  }

  int size() const {
    return storage_.safeSize(nullptr);
  }

  int numAdd() const {
    return numAdd_;
  }

 private:
  using SampleWeightIds = std::tuple<DataType, torch::Tensor, std::vector<int>>;

  SampleWeightIds sample_(int batchsize, const std::string& device) {
    std::unique_lock<std::mutex> lk(mSampler_);

    float sum;
    int size = storage_.safeSize(&sum);
    assert(size >= batchsize);
    // std::cout << "size: "<< size << ", sum: " << sum << std::endl;
    // storage_ [0, size) remains static in the subsequent section

    float segment = sum / batchsize;
    std::uniform_real_distribution<float> dist(0.0, segment);

    std::vector<DataType> samples;
    auto weights = torch::zeros({batchsize}, torch::kFloat32);
    auto weightAcc = weights.accessor<float, 1>();
    std::vector<int> ids(batchsize);

    double accSum = 0;
    int nextIdx = 0;
    float w = 0;
    int id = 0;
    for (int i = 0; i < batchsize; i++) {
      float rand = dist(rng_) + i * segment;
      rand = std::min(sum - (float)0.1, rand);
      // std::cout << "looking for " << i << "th/" << batchsize << " sample" <<
      // std::endl;
      // std::cout << "\ttarget: " << rand << std::endl;

      while (nextIdx <= size) {
        if (accSum > 0 && accSum >= rand) {
          assert(nextIdx >= 1);
          // std::cout << "\tfound: " << nextIdx - 1 << ", " << id << ", " <<
          // accSum << std::endl;
          DataType element = storage_.getElementAndMark(nextIdx - 1);
          samples.push_back(element);
          weightAcc[i] = w;
          ids[i] = id;
          break;
        }

        if (nextIdx == size) {
          std::cout << "nextIdx: " << nextIdx << "/" << size << std::endl;
          std::cout << std::setprecision(10) << "accSum: " << accSum << ", sum: " << sum
                    << ", rand: " << rand << std::endl;
          assert(false);
        }

        w = storage_.getWeight(nextIdx, &id);
        accSum += w;
        ++nextIdx;
      }
    }
    assert((int)samples.size() == batchsize);

    // pop storage if full
    size = storage_.size();
    if (size > capacity_) {
      storage_.blockPop(size - capacity_);
    }

    // safe to unlock, because <samples> contains copys
    lk.unlock();

    weights = weights / sum;
    weights = torch::pow(size * weights, -beta_);
    weights /= weights.max();
    if (device != "cpu") {
      weights = weights.to(torch::Device(device));
    }
    auto batch = DataType::makeBatch(samples, device);
    return std::make_tuple(batch, weights, ids);
  }

  const float alpha_;
  const float beta_;
  const int prefetch_;
  const int capacity_;

  ConcurrentQueue<DataType> storage_;
  std::atomic<int> numAdd_;

  // make sure that sample & update does not overlap
  std::mutex mSampler_;
  std::vector<int> sampledIds_;
  std::queue<std::future<SampleWeightIds>> futures_;

  std::mt19937 rng_;
};

// template class PrioritizedReplay<FFTransition>;
using FFPrioritizedReplay = PrioritizedReplay<FFTransition>;

// template class PrioritizedReplay<RNNTransition>;
using RNNPrioritizedReplay = PrioritizedReplay<RNNTransition>;
}
