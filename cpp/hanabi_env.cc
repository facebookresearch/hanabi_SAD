// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "hanabi_env.h"

rela::TensorDict HanabiEnv::reset() {
  assert(terminated());
  state_ = std::make_unique<hle::HanabiState>(&game_);
  // chance player
  while (state_->CurPlayer() == hle::kChancePlayerId) {
    state_->ApplyRandomChance();
  }
  numStep_ = 0;

  for (int pid = 0; pid < game_.NumPlayers(); ++pid) {
    playerEps_[pid] = epsList_[game_.rng()->operator()() % epsList_.size()];
  }

  if (shuffleColor_) {
    // assert(game_.NumPlayers() == 2);
    int fixColorPlayer = game_.rng()->operator()() % game_.NumPlayers();
    for (int pid = 0; pid < game_.NumPlayers(); ++pid) {
      auto& colorPermute = colorPermutes_[pid];
      auto& invColorPermute = invColorPermutes_[pid];
      colorPermute.clear();
      invColorPermute.clear();
      for (int i = 0; i < game_.NumColors(); ++i) {
        colorPermute.push_back(i);
        invColorPermute.push_back(i);
      }
      if (pid != fixColorPlayer) {
        std::shuffle(colorPermute.begin(), colorPermute.end(), *game_.rng());
        std::sort(invColorPermute.begin(), invColorPermute.end(), [&](int i, int j) {
          return colorPermute[i] < colorPermute[j];
        });
      }
      for (int i = 0; i < (int)colorPermute.size(); ++i) {
        assert(invColorPermute[colorPermute[i]] == i);
      }
    }
  }

  return computeFeatureAndLegalMove(state_);
}

std::tuple<rela::TensorDict, float, bool> HanabiEnv::step(
    const rela::TensorDict& action) {
  assert(!terminated());

  numStep_ += 1;

  float prevScore = state_->Score();

  // perform action for only current player
  int curPlayer = state_->CurPlayer();
  int actionUid = action.at("a")[curPlayer].item<int>();
  hle::HanabiMove move = game_.GetMove(actionUid);
  maybeInversePermuteColor_(move, curPlayer);

  if (!state_->MoveIsLegal(move)) {
    std::cout << "Error: move is not legal" << std::endl;
    std::cout << "UID: " << actionUid << std::endl;
    std::cout << "legal move:" << std::endl;
    std::cout << "numStep: " << numStep_ - 1 << std::endl;

    auto legalMoves = state_->LegalMoves(curPlayer);
    for (auto move : legalMoves) {
      if (shuffleColor_ &&
          move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
        int permColor = colorPermutes_[curPlayer][move.Color()];
        move.SetColor(permColor);
      }
      auto uid = game_.GetMoveUid(move);
      std::cout << "legal_move: " << uid << std::endl;
    }
    assert(false);
  }

  std::unique_ptr<hle::HanabiState> cloneState = nullptr;
  if (sad_) {
    cloneState = std::make_unique<hle::HanabiState>(*state_);
    int greedyActionUid = action.at("greedy_a")[curPlayer].item<int>();
    hle::HanabiMove greedyMove = game_.GetMove(greedyActionUid);
    maybeInversePermuteColor_(greedyMove, curPlayer);

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
    while (state_->CurPlayer() == hle::kChancePlayerId) {
      state_->ApplyRandomChance();
    }
  }

  // std::cout << "score: " << state_->Score() << std::endl;
  auto obs = computeFeatureAndLegalMove(cloneState);
  return std::make_tuple(obs, reward, terminal);
}

rela::TensorDict HanabiEnv::computeFeatureAndLegalMove(
    const std::unique_ptr<hle::HanabiState>& cloneState) {
  std::vector<torch::Tensor> privS;
  // std::vector<torch::Tensor> publS;
  // std::vector<torch::Tensor> superS;
  std::vector<torch::Tensor> legalMove;
  std::vector<torch::Tensor> legalMatrix;
  // auto epsAccessor = eps_.accessor<float, 1>();
  // std::vector<float> eps;
  std::vector<torch::Tensor> ownHand;
  // std::vector<torch::Tensor> ownHandARIn;
  // std::vector<torch::Tensor> allHand;
  // std::vector<torch::Tensor> allHandARIn;

  // std::vector<torch::Tensor> privCardCount;

  for (int i = 0; i < game_.NumPlayers(); ++i) {
    auto obs = hle::HanabiObservation(*state_, i, false);
    std::vector<int> shuffleOrder;
    if (shuffleObs_) {
      // hacked for 2 players
      assert(game_.NumPlayers() == 2);
      // [1] for partner's hand
      int partnerHandSize = obs.Hands()[1].Cards().size();
      for (int i = 0; i < partnerHandSize; ++i) {
        shuffleOrder.push_back(i);
      }
      std::shuffle(shuffleOrder.begin(), shuffleOrder.end(), *game_.rng());
    }

    std::vector<float> vS = obsEncoder_.Encode(
        obs,
        false,
        shuffleOrder,
        shuffleColor_,
        colorPermutes_[i],
        invColorPermutes_[i],
        false);

    if (sad_) {
      assert(cloneState != nullptr);
      auto extraObs = hle::HanabiObservation(*cloneState, i, false);
      std::vector<float> vGreedyAction = obsEncoder_.EncodeLastAction(
          extraObs, shuffleOrder, shuffleColor_, colorPermutes_[i]);
      vS.insert(vS.end(), vGreedyAction.begin(), vGreedyAction.end());
    }

    privS.push_back(torch::tensor(vS));

    {
      auto cheatObs = hle::HanabiObservation(*state_, i, true);
      auto vOwnHand = obsEncoder_.EncodeOwnHandTrinary(cheatObs);
      ownHand.push_back(torch::tensor(vOwnHand));
    }

    // legal moves
    auto legalMoves = state_->LegalMoves(i);
    std::vector<float> moveUids(numAction(), 0);
    // auto moveUids = torch::zeros({numAction()});
    // auto moveAccessor = moveUids.accessor<float, 1>();
    for (auto move : legalMoves) {
      if (shuffleColor_ &&
          // fixColorPlayer_ == i &&
          move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
        int permColor = colorPermutes_[i][move.Color()];
        move.SetColor(permColor);
      }
      auto uid = game_.GetMoveUid(move);
      if (uid >= noOpUid()) {
        std::cout << "Error: legal move id should be < " << numAction() - 1 << std::endl;
        assert(false);
      }
      moveUids[uid] = 1;
    }
    if (legalMoves.size() == 0) {
      moveUids[noOpUid()] = 1;
    }

    legalMove.push_back(torch::tensor(moveUids));
    // epsAccessor[i] = playerEps_[i];
  }

  rela::TensorDict dict = {
      {"priv_s", torch::stack(privS, 0)},
      {"legal_move", torch::stack(legalMove, 0)},
      {"eps", torch::tensor(playerEps_)},
      {"own_hand", torch::stack(ownHand, 0)},
  };

  return dict;
}
