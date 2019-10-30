// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/thread_loop.h"

class HanabiVDNThreadLoop : public rela::ThreadLoop {
 public:
  HanabiVDNThreadLoop(std::shared_ptr<rela::Actor> actor,
                   std::shared_ptr<rela::VectorEnv> env,
                   bool eval)
      : actor_(std::move(actor))
      , env_(std::move(env))
      , eval_(eval) {
    if (eval_) {
      assert(env_->size() == 1);
    }
  }

  void mainLoop() final {
    rela::TensorDict obs = {};
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
  std::shared_ptr<rela::Actor> actor_;
  std::shared_ptr<rela::VectorEnv> env_;
  const bool eval_;
};


class HanabiIQLThreadLoop : public rela::ThreadLoop {
 public:
  HanabiIQLThreadLoop(
      const std::vector<std::shared_ptr<rela::Actor>>& actors,
      std::shared_ptr<rela::VectorEnv> env,
      bool eval)
      : actors_(actors)
      , env_(std::move(env))
      , eval_(eval)
      , logFile_("") {
    if (eval_) {
      assert(env_->size() == 1);
    }
  }

  HanabiIQLThreadLoop(
      const std::vector<std::shared_ptr<rela::Actor>>& actors,
      std::shared_ptr<rela::VectorEnv> env,
      bool eval,
      std::string logFile)
      : actors_(actors)
      , env_(std::move(env))
      , eval_(eval)
      , logFile_(logFile) {
    assert(eval_);
    assert(env_->size() == 1);
  }

  void mainLoop() final {
    std::ofstream file;

    if (!logFile_.empty()) {
      file.open(logFile_);
    }

    rela::TensorDict obs = {};
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

        std::vector<torch::Tensor> actions;
        std::vector<torch::Tensor> greedyAcitons;
        // obs: [batchsize, #player, ...]
        for (int i = 0; i < (int)actors_.size(); ++i) {
          auto input = rela::utils::tensorDictNarrow(obs, 1, i, 1, true);
          if (!logFile_.empty()) {
            logState(file, input);
          }
          // hack ["a"]
          auto reply = actors_[i]->act(input);
          actions.push_back(reply["a"]);
          greedyAcitons.push_back(reply["greedy_a"]);

          if (!logFile_.empty()) {
            logAction(file, reply);
          }
        }

        rela::TensorDict action = {
          {"a", torch::stack(actions, 1)},
          {"greedy_a", torch::stack(greedyAcitons, 1)},
        };
        std::tie(obs, r, t) = env_->step(action);

        if (eval_) {
          continue;
        }

        for (int i = 0; i < (int)actors_.size(); ++i) {
          actors_[i]->setRewardAndTerminal(r, t);
          actors_[i]->postStep();
        }
      }

      // eval only runs for one game
      if (eval_) {
        break;
      }
    }

    if (!logFile_.empty()) {
      file.close();
    }
  }

  void logState(std::ofstream& file, const rela::TensorDict& input) {
    // file << "player: " << i << std::endl;
    for (const auto& kv : input) {
      if (!(kv.first == "s" || kv.first == "legal_move")) {
        continue;
      }
      file << kv.first << ":" << std::endl;
      assert(kv.second.size(0) == 1);
      // std::cout << kv.first << ", " << kv.second.sizes() << std::endl;
      auto s = kv.second[0];
      // std::cout << s.sizes() << std::endl;
      auto accessor = s.accessor<float, 1>();
      for (int j = 0; j < accessor.size(0); ++j) {
        file << accessor[j] << ", ";
      }
      file << std::endl;
    }
  }

  void logAction(std::ofstream& file, const rela::TensorDict& reply) {
    file << "action:" << std::endl;
    file << reply.at("a").item<int>() << std::endl;
    file << "----------" << std::endl;
  }

 private:
  std::vector<std::shared_ptr<rela::Actor>> actors_;
  std::shared_ptr<rela::VectorEnv> env_;
  const bool eval_;
  const std::string logFile_;
};
