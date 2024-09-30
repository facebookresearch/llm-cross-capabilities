# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

CORRELATION_DIR=outputs/correlations
EVALUATOR="gpt"
REFERENCE_NUMBER=2

python correlation/calculate_correlations.py \
    --correlation_file ${CORRELATION_DIR}/${EVALUATOR}_judge_with_${REFERENCE_NUMBER}ref.csv