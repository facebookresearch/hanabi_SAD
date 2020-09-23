# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python contplay_complete.py \
       --save_dir /network/tmp1/badrinaa/hanabi_sad_models/exps/iql_2p_6 \
       --method iql \
       --num_thread 10 \
       --load_learnable_model ../models/iql_2p_6.pthw \
       --load_fixed_model ../models/iql_2p_2.pthw ../models/iql_2p_7.pthw ../models/iql_2p_8.pthw ../models/iql_2p_10.pthw  ../models/iql_2p_12.pthw  \
       --num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 1 \
       --batchsize 128 \
       --burn_in_frames 5000 \
       --replay_buffer_size 131072 \
       --epoch_len 400 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --act_device cuda:0,cuda:1 \
       --shuffle_color 0 \
