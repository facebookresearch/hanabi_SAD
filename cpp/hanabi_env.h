// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"
#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"

#include "rela/env.h"

class HanabiEnv : public rela::Env {
 public:
  HanabiEnv(const std::unordered_map<std::string, std::string>& gameParams,
            int maxLen,
            bool greedyExtra,
            bool verbose)
      : game_(gameParams)
      , obsEncoder_(&game_)
      , state_(nullptr)
      , maxLen_(maxLen)
      , numStep_(0)
      , greedyExtra_(greedyExtra)
      , verbose_(verbose)
  {
    auto params = game_.Parameters();
    if (verbose_) {
      std::cout << "Hanabi game created, with parameters:\n";
      for (const auto& item : params) {
        std::cout << "  " << item.first << "=" << item.second << "\n";
      }
    }
  }

  int featureSize() const {
    int size = obsEncoder_.Shape()[0];
    if (greedyExtra_) {
      size += hanabi_learning_env::LastActionSectionLength(game_);
    }
    return size;
  }

  int numAction() const {
    return game_.MaxMoves() + 1;
  }

  int noOpUid() const {
    return numAction() - 1;
  }

  int handFeatureSize() const {
    return game_.HandSize() * game_.NumColors() * game_.NumRanks();
  }

  // int numPlayer() const {
  //   return game_.NumPlayers();
  // }

  virtual rela::TensorDict reset() final {
    assert(terminated());
    state_ = std::make_unique<hanabi_learning_env::HanabiState>(&game_);
    // chance player
    while (state_->CurPlayer() == hanabi_learning_env::kChancePlayerId) {
      state_->ApplyRandomChance();
    }
    numStep_ = 0;

    return computeFeatureAndLegalMove(state_);
  }

  // return {'obs', 'reward', 'terminal'}
  // action_p0 is a tensor of size 1, representing uid of move
  virtual std::tuple<rela::TensorDict, float, bool> step(
      const rela::TensorDict& action) final {
    assert(!terminated());
    numStep_ += 1;

    float prevScore = state_->Score();

    // perform action for only current player
    int curPlayer = state_->CurPlayer();
    int actionUid = action.at("a")[curPlayer].item<int>();
    hanabi_learning_env::HanabiMove move = game_.GetMove(actionUid);
    if (!state_->MoveIsLegal(move)) {
      std::cout << "Error: move is not legal" << std::endl;
      assert(false);
    }

    std::unique_ptr<hanabi_learning_env::HanabiState> cloneState = nullptr;
    if (greedyExtra_) {
      cloneState = std::make_unique<hanabi_learning_env::HanabiState>(*state_);
      int greedyActionUid = action.at("greedy_a")[curPlayer].item<int>();
      hanabi_learning_env::HanabiMove greedyMove = game_.GetMove(greedyActionUid);
      assert(state_->MoveIsLegal(greedyMove));
      cloneState->ApplyMove(greedyMove);
    }
    state_->ApplyMove(move);

    bool terminal = state_->IsTerminal();
    float reward = state_->Score() - prevScore;
    // forced termination, lose all points
    if (maxLen_ > 0 && numStep_ == maxLen_) {
      terminal = true;
      reward = 0 - prevScore;
    }

    if (!terminal) {
      // chance player
      while (state_->CurPlayer() == hanabi_learning_env::kChancePlayerId) {
        state_->ApplyRandomChance();
      }
    }

    // std::cout << "reward: " << reward << std::endl;
    auto obs = computeFeatureAndLegalMove(cloneState);
    return std::make_tuple(obs, reward, terminal);
  }

  bool terminated() const final {
    if (state_ == nullptr) {
      return true;
    }
    if (maxLen_ <= 0) {
      return state_->IsTerminal();
    } else {
      return state_->IsTerminal() || numStep_ >= maxLen_;
    }
  }

  int getEpisodeReward() const {
    assert(state_ != nullptr);
    return state_->Score();
  }

  std::vector<std::string> deckHistory() const {
    return state_->DeckHistory();
  }

 private:
  rela::TensorDict computeFeatureAndLegalMove(
      const std::unique_ptr<hanabi_learning_env::HanabiState>& cloneState) {
    std::vector<torch::Tensor> s;
    std::vector<torch::Tensor> legalMove;

    for (int i = 0; i < game_.NumPlayers(); ++i) {
      auto obs = hanabi_learning_env::HanabiObservation(*state_, i, false);
      std::vector<float> vS = obsEncoder_.Encode(obs);
      if (greedyExtra_) {
        assert(cloneState != nullptr);
        auto extraObs = hanabi_learning_env::HanabiObservation(*cloneState, i, false);
        std::vector<float> vGreedyAction = obsEncoder_.EncodeLastAction(extraObs);
        vS.insert(vS.end(), vGreedyAction.begin(), vGreedyAction.end());

        // std::vector<float> vRefAction = obsEncoder_.EncodeLastAction(obs);
        // std::cout << "greedy == ref?: " << (vRefAction == vGreedyAction) << std::endl;
      }

      s.push_back(torch::tensor(vS));

      // legal moves
      auto legalMoves = state_->LegalMoves(i);
      auto moveUids = torch::zeros({numAction()});
      auto moveAccessor = moveUids.accessor<float, 1>();
      for (const auto& move : legalMoves) {
        auto uid = game_.GetMoveUid(move);
        if (uid >= noOpUid()) {
          std::cout << "Error: legal move id should be < "
                    << numAction() - 1 << std::endl;
          assert(false);
        }
        moveAccessor[uid] = 1;
      }
      if (legalMoves.size() == 0) {
        moveAccessor[noOpUid()] = 1;
      }

      legalMove.push_back(moveUids);
    }

    rela::TensorDict dict = {
      {"s", torch::stack(s, 0)},
      {"legal_move", torch::stack(legalMove, 0)},
    };

    return dict;
  }

  const hanabi_learning_env::HanabiGame game_;
  const hanabi_learning_env::CanonicalObservationEncoder obsEncoder_;
  std::unique_ptr<hanabi_learning_env::HanabiState> state_;
  const int maxLen_;
  int numStep_;

  const bool greedyExtra_;
  const bool verbose_;
};
