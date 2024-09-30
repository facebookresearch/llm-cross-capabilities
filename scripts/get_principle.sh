# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

MODEL_VERSION="gpt-4o-2024-05-13"
SCORE_DIR=outputs/scores
PRINCIPLE_DIR=outputs/principles
CAPABILITY="Reasoning"

python principle_prompting/generate_principles.py \
    --response=${MODEL_VERSION}_response \
    --score_file=${SCORE_DIR}/${MODEL_VERSION}_response_gpt_score.csv \
    --capability=${CAPABILITY} \
    --save_path=${PRINCIPLE_DIR}/${MODEL_VERSION}_${CAPABILITY}.txt
