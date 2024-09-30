# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm

from gpt import GPT
from utils import get_user_prompt


def is_file_empty(file_path):
    """
    Checks if a file exists and is empty.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists and is empty, False otherwise.
    """
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def parse_principles(model_response, capability):
    """
    Extracts the principles from the model's response.

    Parameters:
        model_response (str): The model's response containing the principles.
        capability (str): The capability for which the principle is generated.

    Returns:
        str: The extracted principle text.
    """
    start_phrase = f"## Principles for Prompts related to {capability}"
    end_phrase = "[END of Principles]"

    start_match = re.search(re.escape(start_phrase), model_response)
    end_match = re.search(re.escape(end_phrase), model_response)

    if start_match and end_match:
        start_index = start_match.start()
        end_index = end_match.end()
        return model_response[start_index:end_index].strip()
    else:
        return ""


def generate_principle(
    instance, response_key, cur_principle, index, model, print_principle=True
):
    """
    Generates principles using GPT with the current instance.

    Parameters:
        instance (dict): The current data instance.
        response_key (str): The key for the model's response in the instance.
        cur_principle (str): The current accumulated principles.
        index (int): The index of the current iteration.
        model (str): The LLM used to generate the principles.

    Returns:
        str: The new generated principles.
    """
    user_prompt = get_user_prompt(
        instance=instance,
        response=response_key,
        cur_principle=cur_principle,
        idx=index,
        file_url=(
            instance["attached_file"]
            if "long context" in instance["capability"].lower()
            else None
        ),
    )

    # Handle image capabilities
    if "image" in instance["capability"].lower():
        file_url = instance["attached_file"]
    else:
        file_url = None

    # Get the model's response
    model_response = model.get_response(user_prompt=user_prompt, image_url=file_url)

    if print_principle:
        print("Input Prompt:")
        print(user_prompt)
        print("Current principles:")
        print(model_response)

    return model_response


def main(args):

    # Initialize the model
    model = GPT()

    # Ensure the directory for save_path exists
    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(args.score_file)
    df = df[df["capability"] == args.capability]
    assert (
        len(df) == 100
    ), f"Expected 100 instances for capability {args.capability}, found {len(df)}"

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Initialize the current principle
    cur_principle = "[No principles yet. Please generate it from scratch.]"
    cur_index = 1

    # Iterate over the DataFrame rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating guidelines"):
        max_retries = 5
        retry_cnt = 0
        while retry_cnt < max_retries:
            model_response = generate_principle(
                instance=row,
                response_key=args.response,
                cur_principle=cur_principle,
                index=cur_index,
                model=model,
                print_principle=True,
            )
            new_principle = parse_principles(model_response, args.capability)
            if new_principle != "":
                cur_principle = new_principle
                # Overwrite the file each time with the latest principles
                with open(args.save_path, "w", encoding="utf-8") as f_save:
                    f_save.write(cur_principle)
                break
            else:
                retry_cnt += 1
                print(f"Retrying ({retry_cnt}/{max_retries})...")
        if retry_cnt == max_retries:
            raise Exception(
                f"Cannot generate principles for prompt_id {row['prompt_id']} after {max_retries} retries."
            )
        cur_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate principles based on model responses and scores"
    )

    parser.add_argument(
        "--response", type=str, required=True, help="Model response identifier"
    )
    parser.add_argument(
        "--score_file",
        type=str,
        required=True,
        help="File path to the CSV file containing responses and scores",
    )
    parser.add_argument(
        "--capability",
        type=str,
        required=True,
        help="The capability for which to generate principles",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="File path to save the generated principles",
    )
    args = parser.parse_args()

    main(args)
