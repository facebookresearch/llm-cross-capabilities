# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

MODEL_VERSION="gpt-4o-2024-05-13"
RESPONSE_DIR=outputs/responses
SCORE_DIR=outputs/scores
EVALUATOR=gpt

python evaluation/evaluate_response.py \
    --response=${MODEL_VERSION}_response \
    --response_file=${RESPONSE_DIR}/${MODEL_VERSION}.csv \
    --save_path=${SCORE_DIR}/${MODEL_VERSION}_response_${EVALUATOR}_score.csv \
    --evaluator=${EVALUATOR}