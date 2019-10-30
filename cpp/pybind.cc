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
  py::class_<HanabiEnv, rela::Env, std::shared_ptr<HanabiEnv>>(m, "HanabiEnv")
      .def(py::init<const std::unordered_map<std::string, std::string>&,
                    int,
                     bool,
                    bool>())
      .def("feature_size", &HanabiEnv::featureSize)
      .def("num_action", &HanabiEnv::numAction)
      .def("reset", &HanabiEnv::reset)
      .def("step", &HanabiEnv::step)
      .def("terminated", &HanabiEnv::terminated)
      .def("get_episode_reward", &HanabiEnv::getEpisodeReward)
      .def("hand_feature_size", &HanabiEnv::handFeatureSize)
      .def("deck_history", &HanabiEnv::deckHistory);

  py::class_<HanabiVDNThreadLoop,
             rela::ThreadLoop,
             std::shared_ptr<HanabiVDNThreadLoop>>(m, "HanabiVDNThreadLoop")
      .def(py::init<
           std::shared_ptr<rela::Actor>,
           std::shared_ptr<rela::VectorEnv>,
           bool>());

  py::class_<HanabiIQLThreadLoop,
             rela::ThreadLoop,
             std::shared_ptr<HanabiIQLThreadLoop>>(m, "HanabiIQLThreadLoop")
      .def(py::init<
           const std::vector<std::shared_ptr<rela::Actor>>&,
           std::shared_ptr<rela::VectorEnv>,
           bool>())
      .def(py::init<
           const std::vector<std::shared_ptr<rela::Actor>>&,
           std::shared_ptr<rela::VectorEnv>,
           bool,
           std::string>());
}
