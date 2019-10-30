# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tube = os.path.join(root, "build", "rela")
    if tube not in sys.path:
        sys.path.append(tube)

    hanalearn = os.path.join(root, "build")
    if hanalearn not in sys.path:
        sys.path.append(hanalearn)
