// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rela/env.h"

namespace hle = hanabi_learning_env;

class HanabiEnv : public rela::Env {
 public:
  HanabiEnv(
      const std::unordered_map<std::string, std::string>& gameParams,
      const std::vector<float>& epsList,
      int maxLen,
      bool sad,
      bool shuffleObs,
      bool shuffleColor,
      bool verbose)
      : game_(gameParams)
      , obsEncoder_(&game_)
      , state_(nullptr)
      , epsList_(epsList)
      , maxLen_(maxLen)
      , sad_(sad)
      , shuffleObs_(shuffleObs)
      , shuffleColor_(shuffleColor)
      , verbose_(verbose)
      , playerEps_(game_.NumPlayers())
      , numStep_(0)
      , colorPermutes_(game_.NumPlayers())
      , invColorPermutes_(game_.NumPlayers())
      , lastScore_(-1) {
    auto params = game_.Parameters();
    if (verbose_) {
      std::cout << "Hanabi game created, with parameters:\n";
      for (const auto& item : params) {
        std::cout << "  " << item.first << "=" << item.second << "\n";
      }
    }
  }

  virtual ~HanabiEnv() {
  }

  int featureSize() const {
    int size = obsEncoder_.Shape()[0];
    if (sad_) {
      size += hle::LastActionSectionLength(game_);
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

  virtual rela::TensorDict reset() override;

  // return {'obs', 'reward', 'terminal'}
  // action_p0 is a tensor of size 1, representing uid of move
  virtual std::tuple<rela::TensorDict, float, bool> step(
      const rela::TensorDict& action) override;

  bool terminated() const final {
    if (state_ == nullptr) {
      return true;
    }

    bool term = false;
    if (maxLen_ <= 0) {
      term = state_->IsTerminal();
    } else {
      term = state_->IsTerminal() || numStep_ >= maxLen_;
    }
    if (term) {
      lastScore_ = state_->Score();
    }
    return term;
  }

  int getCurrentPlayer() const {
    assert(state_ != nullptr);
    return state_->CurPlayer();
  }

  bool moveIsLegal(int actionUid) const {
    hle::HanabiMove move = game_.GetMove(actionUid);
    return state_->MoveIsLegal(move);
  }

  int lastScore() const {
    return lastScore_;
  }

  std::vector<std::string> deckHistory() const {
    return state_->DeckHistory();
  }

  const hle::HanabiState& getHanabiState() const {
    assert(state_ != nullptr);
    return *state_;
  }

  int getScore() const {
    return state_->Score();
  }

  int getLife() const {
    return state_->LifeTokens();
  }

  int getInfo() const {
    return state_->InformationTokens();
  }

  std::vector<int> getFireworks() const {
    return state_->Fireworks();
  }

 protected:
  bool maybeInversePermuteColor_(hle::HanabiMove& move, int curPlayer) {
    if (shuffleColor_ && move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
      int realColor = invColorPermutes_[curPlayer][move.Color()];
      move.SetColor(realColor);
      return true;
    } else {
      return false;
    }
  }

  rela::TensorDict computeFeatureAndLegalMove(
      const std::unique_ptr<hle::HanabiState>& cloneState);

  const hle::HanabiGame game_;
  const hle::CanonicalObservationEncoder obsEncoder_;
  std::unique_ptr<hle::HanabiState> state_;
  const std::vector<float> epsList_;
  const int maxLen_;
  const bool sad_;
  const bool shuffleObs_;
  const bool shuffleColor_;
  const bool verbose_;

  std::vector<float> playerEps_;

  int numStep_;
  std::vector<std::vector<int>> colorPermutes_;
  std::vector<std::vector<int>> invColorPermutes_;

  mutable int lastScore_;
};
