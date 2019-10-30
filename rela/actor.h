// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/types.h"

namespace rela {

class Actor {
 public:
  Actor() = default;

  virtual ~Actor() {
  }

  virtual TensorDict act(TensorDict& obs) = 0;

  virtual void setRewardAndTerminal(torch::Tensor& r, torch::Tensor& t) = 0;

  virtual void postStep() = 0;
};
}
