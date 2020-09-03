// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include "rela/transition.h"
#include "rela/utils.h"

using namespace rela;

FFTransition FFTransition::index(int i) const {
  FFTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.terminal = terminal[i];
  element.bootstrap = bootstrap[i];

  for (auto& name2tensor : nextObs) {
    element.nextObs.insert({name2tensor.first, name2tensor.second[i]});
  }

  return element;
}

FFTransition FFTransition::padLike() const {
  FFTransition pad;

  pad.obs = tensor_dict::zerosLike(obs);
  pad.action = tensor_dict::zerosLike(action);
  pad.reward = torch::zeros_like(reward);
  pad.terminal = torch::ones_like(terminal);
  pad.bootstrap = torch::zeros_like(bootstrap);
  pad.nextObs = tensor_dict::zerosLike(nextObs);

  return pad;
}

std::vector<torch::jit::IValue> FFTransition::toVectorIValue(
    const torch::Device& device) const {
  std::vector<torch::jit::IValue> vec;
  vec.push_back(tensor_dict::toIValue(obs, device));
  vec.push_back(tensor_dict::toIValue(action, device));
  vec.push_back(reward.to(device));
  vec.push_back(terminal.to(device));
  vec.push_back(bootstrap.to(device));
  vec.push_back(tensor_dict::toIValue(nextObs, device));
  return vec;
}

TensorDict FFTransition::toDict() {
  auto dict = obs;
  for (auto& kv : nextObs) {
    dict["next_" + kv.first] = kv.second;
  }

  for (auto& kv : action) {
    auto ret = dict.emplace(kv.first, kv.second);
    assert(ret.second);
  }

  auto ret = dict.emplace("reward", reward);
  assert(ret.second);
  ret = dict.emplace("terminal", terminal);
  assert(ret.second);
  ret = dict.emplace("bootstrap", bootstrap);
  assert(ret.second);
  return dict;
}

RNNTransition::RNNTransition(
    const std::vector<FFTransition>& transitions, TensorDict h0, torch::Tensor seqLen)
    : h0(h0)
    , seqLen(seqLen) {
  std::vector<TensorDict> obsVec;
  std::vector<TensorDict> actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    obsVec.push_back(transitions[i].obs);
    actionVec.push_back(transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    terminalVec.push_back(transitions[i].terminal);
    bootstrapVec.push_back(transitions[i].bootstrap);
  }

  obs = tensor_dict::stack(obsVec, 0);
  action = tensor_dict::stack(actionVec, 0);
  reward = torch::stack(rewardVec, 0);
  terminal = torch::stack(terminalVec, 0);
  bootstrap = torch::stack(bootstrapVec, 0);
}

RNNTransition RNNTransition::index(int i) const {
  RNNTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : h0) {
    auto t = name2tensor.second.narrow(1, i, 1).squeeze(1);
    element.h0.insert({name2tensor.first, t});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.terminal = terminal[i];
  element.bootstrap = bootstrap[i];
  element.seqLen = seqLen[i];
  return element;
}

FFTransition FFTransition::makeBatch(
    const std::vector<FFTransition>& transitions, const std::string& device) {
  std::vector<TensorDict> obsVec;
  std::vector<TensorDict> actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;
  std::vector<TensorDict> nextObsVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    obsVec.push_back(transitions[i].obs);
    actionVec.push_back(transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    terminalVec.push_back(transitions[i].terminal);
    bootstrapVec.push_back(transitions[i].bootstrap);
    nextObsVec.push_back(transitions[i].nextObs);
  }

  FFTransition batch;
  batch.obs = tensor_dict::stack(obsVec, 0);
  batch.action = tensor_dict::stack(actionVec, 0);
  batch.reward = torch::stack(rewardVec, 0);
  batch.terminal = torch::stack(terminalVec, 0);
  batch.bootstrap = torch::stack(bootstrapVec, 0);
  batch.nextObs = tensor_dict::stack(nextObsVec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
    batch.obs = tensor_dict::apply(batch.obs, toDevice);
    batch.action = tensor_dict::apply(batch.action, toDevice);
    batch.reward = batch.reward.to(d);
    batch.terminal = batch.terminal.to(d);
    batch.bootstrap = batch.bootstrap.to(d);
    batch.nextObs = tensor_dict::apply(batch.nextObs, toDevice);
  }

  return batch;
}

RNNTransition RNNTransition::makeBatch(
    const std::vector<RNNTransition>& transitions, const std::string& device) {
  std::vector<TensorDict> obsVec;
  // TensorVecDict h0Vec;
  std::vector<TensorDict> actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;
  std::vector<torch::Tensor> seqLenVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    obsVec.push_back(transitions[i].obs);
    // utils::tensorVecDictAppend(h0Vec, transitions[i].h0);
    actionVec.push_back(transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    terminalVec.push_back(transitions[i].terminal);
    bootstrapVec.push_back(transitions[i].bootstrap);
    seqLenVec.push_back(transitions[i].seqLen);
  }

  RNNTransition batch;
  batch.obs = tensor_dict::stack(obsVec, 1);
  // batch.h0 = tensor_dict::stack(h0Vec, 1);  // 1 is batch for rnn hid
  batch.action = tensor_dict::stack(actionVec, 1);
  batch.reward = torch::stack(rewardVec, 1);
  batch.terminal = torch::stack(terminalVec, 1);
  batch.bootstrap = torch::stack(bootstrapVec, 1);
  batch.seqLen = torch::stack(seqLenVec, 0); //.squeeze(1);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
    batch.obs = tensor_dict::apply(batch.obs, toDevice);
    // batch.h0 = tensor_dict::apply(batch.h0, toDevice);
    batch.action = tensor_dict::apply(batch.action, toDevice);
    batch.reward = batch.reward.to(d);
    batch.terminal = batch.terminal.to(d);
    batch.bootstrap = batch.bootstrap.to(d);
    batch.seqLen = batch.seqLen.to(d);
  }

  return batch;
}
