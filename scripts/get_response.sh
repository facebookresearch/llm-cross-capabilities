# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

### GPT-4o
RESPONSE_DIR=outputs/responses
MODEL_NAME="gpt"
MODEL_VERSION="gpt-4o-2024-05-13"

python generate_response/generate.py \
    --save_path="${RESPONSE_DIR}/${MODEL_VERSION}.csv" \
    --model="${MODEL_NAME}" \
    --model_version="${MODEL_VERSION}" \
    --enable_code_interpreter \

### Claude 3.5 Sonnet
# RESPONSE_DIR=outputs/responses
# MODEL_NAME="claude"
# MODEL_VERSION="claude-3-5-sonnet-20240620"

# python generate_response/generate.py \
#     --save_path="${RESPONSE_DIR}/${MODEL_VERSION}.csv" \
#     --model="${MODEL_NAME}" \
#     --model_version="${MODEL_VERSION}" \

### Gemini 1.5 Pro Exp
# RESPONSE_DIR=outputs/responses
# MODEL_NAME="gemini"
# MODEL_VERSION="gemini-1.5-pro-exp-0801"

# python generate_response/generate.py \
#     --save_path="${RESPONSE_DIR}/${MODEL_VERSION}.csv" \
#     --model="${MODEL_NAME}" \
#     --model_version="${MODEL_VERSION}" \

### Reka Core
# RESPONSE_DIR=outputs/responses
# MODEL_NAME="reka"
# MODEL_VERSION="reka-core-20240722"

# python generate_response/generate.py \
#     --save_path="${RESPONSE_DIR}/${MODEL_VERSION}.csv" \
#     --model="${MODEL_NAME}" \
#     --model_version="${MODEL_VERSION}" \
