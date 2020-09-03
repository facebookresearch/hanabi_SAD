// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rela/context.h"
#include "rela/env.h"
#include "rela/prioritized_replay.h"
#include "rela/r2d2_actor.h"
#include "rela/thread_loop.h"
#include "rela/transition.h"

namespace py = pybind11;
using namespace rela;

PYBIND11_MODULE(rela, m) {
  py::class_<FFTransition, std::shared_ptr<FFTransition>>(m, "FFTransition")
      .def_readwrite("obs", &FFTransition::obs)
      .def_readwrite("action", &FFTransition::action)
      .def_readwrite("reward", &FFTransition::reward)
      .def_readwrite("terminal", &FFTransition::terminal)
      .def_readwrite("bootstrap", &FFTransition::bootstrap)
      .def_readwrite("next_obs", &FFTransition::nextObs);

  py::class_<RNNTransition, std::shared_ptr<RNNTransition>>(m, "RNNTransition")
      .def_readwrite("obs", &RNNTransition::obs)
      .def_readwrite("h0", &RNNTransition::h0)
      .def_readwrite("action", &RNNTransition::action)
      .def_readwrite("reward", &RNNTransition::reward)
      .def_readwrite("terminal", &RNNTransition::terminal)
      .def_readwrite("bootstrap", &RNNTransition::bootstrap)
      .def_readwrite("seq_len", &RNNTransition::seqLen);

  // py::class_<FFPrioritizedReplay, std::shared_ptr<FFPrioritizedReplay>>(
  //     m, "FFPrioritizedReplay")
  //     .def(py::init<int,    // capacity,
  //                   int,    // seed,
  //                   float,  // alpha, priority exponent
  //                   float,  // beta, importance sampling exponent
  //                   int>())
  //     .def("size", &FFPrioritizedReplay::size)
  //     .def("num_add", &FFPrioritizedReplay::numAdd)
  //     .def("sample", &FFPrioritizedReplay::sample)
  //     .def("update_priority", &FFPrioritizedReplay::updatePriority);

  py::class_<RNNPrioritizedReplay, std::shared_ptr<RNNPrioritizedReplay>>(
      m, "RNNPrioritizedReplay")
      .def(py::init<
           int,    // capacity,
           int,    // seed,
           float,  // alpha, priority exponent
           float,  // beta, importance sampling exponent
           int>())
      .def("size", &RNNPrioritizedReplay::size)
      .def("num_add", &RNNPrioritizedReplay::numAdd)
      .def("sample", &RNNPrioritizedReplay::sample)
      .def("update_priority", &RNNPrioritizedReplay::updatePriority)
      .def("get", &RNNPrioritizedReplay::get);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("push_env_thread", &Context::pushThreadLoop, py::keep_alive<1, 2>())
      .def("start", &Context::start)
      .def("pause", &Context::pause)
      .def("resume", &Context::resume)
      .def("terminate", &Context::terminate)
      .def("terminated", &Context::terminated);

  py::class_<R2D2Actor, std::shared_ptr<R2D2Actor>>(m, "R2D2Actor")
      .def(py::init<
           std::shared_ptr<BatchRunner>,                   // runner
           int,                                            // multiStep
           int,                                            // batchsize
           float,                                          // gamma
           float,                                          // eta
           int,                                            // seqLen
           int,                                            // numPlayer
           std::shared_ptr<RNNPrioritizedReplay>>())       // replayBuffer
      .def(py::init<std::shared_ptr<BatchRunner>, int>())  // evaluation mode
      .def("num_act", &R2D2Actor::numAct)
      ;

  py::class_<BatchRunner, std::shared_ptr<BatchRunner>>(m, "BatchRunner")
      .def(py::init<py::object, const std::string&, int, const std::vector<std::string>&>())
      .def("start", &BatchRunner::start)
      .def("stop", &BatchRunner::stop)
      .def("update_model", &BatchRunner::updateModel)
      ;

  m.def("aggregate_priority", &aggregatePriority);
}
