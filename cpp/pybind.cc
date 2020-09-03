// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <pybind11/pybind11.h>

#include "cpp/hanabi_env.h"
#include "cpp/thread_loop.h"

namespace py = pybind11;

PYBIND11_MODULE(hanalearn, m) {
  py::class_<HanabiEnv, std::shared_ptr<HanabiEnv>>(m, "HanabiEnv")
      .def(py::init<
           const std::unordered_map<std::string, std::string>&,
           const std::vector<float>&,
           int,   // maxLen
           bool,  // sad
           bool,  // shuffleObs
           bool,  // shuffleColor
           bool>())
      .def("feature_size", &HanabiEnv::featureSize)
      .def("num_action", &HanabiEnv::numAction)
      .def("reset", &HanabiEnv::reset)
      .def("step", &HanabiEnv::step)
      .def("terminated", &HanabiEnv::terminated)
      .def("get_current_player", &HanabiEnv::getCurrentPlayer)
      .def("move_is_legal", &HanabiEnv::moveIsLegal)
      .def("last_score", &HanabiEnv::lastScore)
      .def("hand_feature_size", &HanabiEnv::handFeatureSize)
      .def("deck_history", &HanabiEnv::deckHistory)
      .def("get_score", &HanabiEnv::getScore)
      .def("get_life", &HanabiEnv::getLife)
      .def("get_info", &HanabiEnv::getInfo)
      .def("get_fireworks", &HanabiEnv::getFireworks)
      ;

  py::class_<HanabiVecEnv, std::shared_ptr<HanabiVecEnv>>(m, "HanabiVecEnv")
      .def(py::init<>())
      .def("append", &HanabiVecEnv::append, py::keep_alive<1, 2>())
      ;

  py::class_<HanabiThreadLoop, rela::ThreadLoop, std::shared_ptr<HanabiThreadLoop>>(
      m, "HanabiThreadLoop")
      .def(py::init<
           std::shared_ptr<rela::R2D2Actor>,
           std::shared_ptr<HanabiVecEnv>,
           bool>())
      .def(py::init<
           std::vector<std::shared_ptr<rela::R2D2Actor>>,
           std::shared_ptr<HanabiVecEnv>,
           bool>())
      ;
}
