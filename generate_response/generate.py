# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import sys
import csv
import argparse
from tqdm import tqdm
from datasets import load_dataset

from models.reka import REKA
from models.claude import Claude
from models.gemini import Gemini
from models.gpt import GPT, Tool_GPT
from models.utils import get_system_prompt, get_user_prompt


def is_file_empty(file_path):
    """Check if the file exists and is empty."""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def gpt_response(instance, args, model, tool_model, print_response=True):
    """
    Generate a response for a given instance using the specified model.

    Args:
        instance (dict): The data instance containing the prompt and other info.
        args (Namespace): The command-line arguments.
        model: The LLM used to generate responses.
        tool_model: The LLM supports tool use function (e.g., code interpreter), or None.

    Returns:
        dict: The instance updated with the model's response.
    """
    model_name = args.model

    # Determine if an attached file should be included in the user prompt
    if "long context" in instance["capability"].lower():
        # For long context, include the attached file URL in the user prompt
        file_url_in_prompt = instance["attached_file"]
        file_url = None
    else:
        file_url_in_prompt = None
        # For image and tool use, send the attached file URL directly to the API
        if instance["attached_file"] != "":
            file_url = instance["attached_file"]
        else:
            file_url = None

    # Generate the user prompt
    user_prompt = get_user_prompt(
        prompt=instance["prompt"], file_url=file_url_in_prompt, model=args.model
    )

    # Handle tool use-related capabilities
    if "tool use" in instance["capability"].lower():
        # Capabilities involving both tool use and reasoning/coding require code interpreter
        if "&" in instance["capability"].lower():
            if tool_model is not None:
                cur_response = tool_model.get_response(
                    user_prompt=user_prompt, file_url=file_url
                )
            else:
                cur_response = ""
        else:
            # Capabilities requiring web browsing, which none of the models support
            cur_response = ""
    else:
        # For models like Claude that require a system prompt during inference
        if args.model == "claude":
            system_prompt = get_system_prompt()
            cur_response = model.get_response(
                user_prompt=user_prompt, system_prompt=system_prompt, file_url=file_url
            )
        else:
            cur_response = model.get_response(
                user_prompt=user_prompt, file_url=file_url
            )

    if print_response:
        print(f"{model_name} response:")
        print(cur_response)

    instance[f"{args.model_version}_response"] = cur_response
    return instance


def main(args):

    # Load the CrossEval benchmark dataset
    dataset = load_dataset("MingZhong/crosseval", split="test")

    # Initialize the specified model
    if args.model == "gpt":
        cur_model = GPT(args.model_version, get_system_prompt())
    elif args.model == "claude":
        cur_model = Claude(args.model_version)
    elif args.model == "gemini":
        cur_model = Gemini(args.model_version)
    elif args.model == "reka":
        cur_model = REKA(args.model_version)
    else:
        print(f"Model {args.model} not supported.")
        sys.exit(1)

    # Initialize the tool model if code interpreter is enabled
    if args.enable_code_interpreter:
        tool_model = Tool_GPT(
            args.model_version, get_system_prompt(args.enable_code_interpreter)
        )
    else:
        tool_model = None

    # Ensure the directory for save_path exists
    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Read existing output file to determine which prompt_ids have been processed
    processed_ids = set()
    if os.path.exists(args.save_path):
        with open(args.save_path, mode="r", newline="") as outfile:
            reader = csv.DictReader(outfile)
            processed_ids = {row["prompt_id"] for row in reader}

    # Determine fieldnames: existing fields plus the new response field
    response_name = f"{args.model_version}_response"
    fieldnames = list(dataset.features.keys()) + [response_name]

    # Open the output CSV file in append mode
    with open(args.save_path, mode="a", newline="", buffering=1) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # Write header only if the file is new or empty
        if is_file_empty(args.save_path):
            writer.writeheader()

        # Generate response for each instance
        for instance in tqdm(dataset, desc="Processing rows"):
            cur_prompt_id = instance.get("prompt_id")

            if cur_prompt_id not in processed_ids:  # Process only new rows
                try:
                    updated_instance = gpt_response(
                        instance, args, cur_model, tool_model, print_response=True
                    )
                    writer.writerow(updated_instance)
                    processed_ids.add(cur_prompt_id)
                except Exception as e:
                    print(
                        f"Error processing instance with prompt_id {instance['prompt_id']}: {e}"
                    )
                    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model responses on the CrossEval benchmark"
    )

    parser.add_argument(
        "--save_path", type=str, help="File path to store output responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt",
        choices=["gpt", "claude", "gemini", "reka"],
        help="LLM used to generate responses",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Exact version used in the generation",
    )
    parser.add_argument(
        "--enable_code_interpreter",
        action="store_true",
        help="Enable code interpreter for capabilities involving tool use (currently only supported for GPT)",
    )
    args = parser.parse_args()

    main(args)
