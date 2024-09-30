# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import re
import csv
import argparse
import pandas as pd
from tqdm import tqdm

from evaluator import GPT_Evaluator, Tool_Evaluator, Claude_Evaluator
from utils import get_system_prompt, get_user_prompt_with_3ref


def is_file_empty(file_path):
    """
    Checks if a file exists and is empty.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists and is empty, False otherwise.
    """
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def parse_score(text):
    """
    Parses the evaluation score from the given explanation.

    Parameters:
        text (str): The text containing the score.

    Returns:
        float or None: The extracted score as a float, or None if not found.
    """
    # Regular expression to find scores in the format "score: [number]/5", case-insensitive
    match = re.search(
        r"evaluation score[:\*\s]*score:\s*\**(\d+(\.\d+)?)/5\**", text, re.IGNORECASE
    )
    if match:
        score_str = match.group(1)
        try:
            score = float(score_str)
            return score
        except ValueError:
            return None
    return None


def get_judge(
    instance, response, evaluator, tool_evaluator, evaluator_type, print_evaluation=True
):
    """
    Evaluates a model's response and updates the instance with the rating and explanation.

    Parameters:
        instance (dict): The current instance, including the attached file, user prompt, and reference examples.
        response (str): The key for the model's response in the instance.
        evaluator: The evaluator object (GPT_Evaluator or Claude_Evaluator).
        tool_evaluator: The tool evaluator object (Tool_Evaluator), if any.
        evaluator_type (str): The type of evaluator ('gpt' or 'claude').

    Returns:
        dict: The updated instance with the evaluation results.
    """
    if instance[response] == "" or instance[response] == "nan":
        instance[f"{response}_rating"] = ""
        instance[f"{response}_explanation"] = ""
        return instance

    # Generate the user prompt for evaluation
    user_prompt = get_user_prompt_with_3ref(
        prompt=instance["prompt"],
        response=instance[response],
        instance=instance,
        evaluator=evaluator_type,
    )

    # Handle cases where the model failed to provide an answer
    if "Failed to get the response" in user_prompt:
        user_prompt += " The model may not provide an answer due to potential safety concerns or issues with recitation, resulting in a lack of response (e.g., Failed to get the response). In such cases, evaluate whether there is a risk, and if there isn't, you can give a score of 1."
    if len(instance[response].split()) < 2:
        user_prompt += " The model may provide an overly brief response, such as just a placeholder. In cases where this completely fails to meet user requirements, you can assign a score of 1."

    # Determine if an attached file should be included
    if instance["attached_file"] != "" and (
        "long context" not in instance["capability"].lower()
    ):
        file_url = instance["attached_file"]
        # For Claude, ignore file URL in tool use
        if "tool use" in instance["capability"].lower() and evaluator_type == "claude":
            file_url = None
    else:
        file_url = None

    max_retries = 5
    retry_cnt = 0
    while retry_cnt < max_retries:
        if retry_cnt == 1:
            # Invalid rating
            if "The rating should be on a scale from 1 to 5." not in user_prompt:
                if "tool use" in instance["capability"].lower():
                    # No attached file
                    user_prompt += " If it doesn't provide the attached file, evaluate the response based on the reference examples."
                else:
                    # Refused to provide the evaluation
                    user_prompt += " Focus on providing your evaluation without reproducing or paraphrasing any attached text."
                user_prompt = user_prompt.strip()
        if "tool use" in instance["capability"].lower():
            if evaluator_type == "gpt":
                # Enable code interpreter for GPT
                if tool_evaluator is not None:
                    explanation = tool_evaluator.evaluate(
                        user_prompt=user_prompt, file_url=file_url
                    )
                else:
                    file_url = None
                    explanation = evaluator.evaluate(
                        user_prompt=user_prompt, image_url=file_url
                    )
                rating = parse_score(explanation)
            else:
                # For Claude, input system prompt during the inference
                system_prompt = get_system_prompt(use_tool=True)
                explanation = evaluator.evaluate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    image_url=file_url,
                )
                rating = parse_score(explanation)
        else:
            if evaluator_type == "gpt":
                explanation = evaluator.evaluate(
                    user_prompt=user_prompt, image_url=file_url
                )
                rating = parse_score(explanation)
            else:
                system_prompt = get_system_prompt()
                explanation = evaluator.evaluate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    image_url=file_url,
                )
                rating = parse_score(explanation)
        if print_evaluation:
            print("User Prompt:")
            print(user_prompt)
            print("Evaluation:")
            print(explanation)
            print(f"\nEvaluator: {evaluator_type}")
        if isinstance(rating, float):
            if 1 <= rating <= 5:
                instance[f"{response}_rating"] = rating
                instance[f"{response}_explanation"] = explanation
                break
            else:
                user_prompt += " The rating should be on a scale from 1 to 5."
        retry_cnt += 1
    if retry_cnt == max_retries:
        raise Exception(
            f"Can't get evaluation scores for {response} in {instance['prompt_id']}!"
        )
    return instance


def scale_score(score):
    """
    Rescales the score from a 1-5 scale to a 1-100 scale.

    Parameters:
        score (float): The original score on a 1-5 scale.

    Returns:
        float: The rescaled score on a 1-100 scale.
    """
    return (score - 1) * 24.75 + 1


def calculate_average_scores(df, response):
    """
    Calculates the average scores for each capability.

    Parameters:
        df (DataFrame): The DataFrame containing the evaluation results.
        response (str): The response key to calculate scores for.

    Returns:
        dict: A dictionary with capabilities as keys and average scores as values.
    """
    results = {}

    capabilities = df["capability"].unique()

    for capability in capabilities:
        cap_data = df[df["capability"] == capability]
        full_set_data = cap_data[
            pd.to_numeric(cap_data[f"{response}_rating"], errors="coerce").notna()
        ]
        full_set_avg_score = full_set_data[f"{response}_rating"].astype(float).mean()
        results[capability] = full_set_avg_score

    return results


def print_scores(file_path):
    """
    Prints the average scores for each capability.

    Parameters:
        file_path (str): The path to the CSV file containing evaluation results.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    response = os.path.basename(file_path)[
        : os.path.basename(file_path).find("response") + len("response")
    ]

    # Apply the scaling to the response scores, ignoring NaNs
    df[f"{response}_rating"] = pd.to_numeric(
        df[f"{response}_rating"], errors="coerce"
    ).apply(lambda x: scale_score(x) if pd.notna(x) else x)

    # Calculate the average scores
    average_scores = calculate_average_scores(df, response)

    # Separate individual and cross capabilities
    individual_capabilities = {k: v for k, v in average_scores.items() if "&" not in k}
    cross_capabilities = {k: v for k, v in average_scores.items() if "&" in k}

    # Convert results to DataFrames for better readability
    individual_capabilities_df = pd.DataFrame(
        individual_capabilities.items(), columns=["Capability", "Average Score"]
    )
    cross_capabilities_df = pd.DataFrame(
        cross_capabilities.items(), columns=["Capability", "Average Score"]
    )

    # Display the results
    print("Individual Capabilities:\n", individual_capabilities_df.to_string(index=False))
    print("\nCross Capabilities:\n", cross_capabilities_df.to_string(index=False))


def main(args):

    # Initialize evaluator
    if args.evaluator == "gpt":
        evaluator = GPT_Evaluator(get_system_prompt())
    else:
        evaluator = Claude_Evaluator()
    if args.enable_code_interpreter:
        tool_evaluator = Tool_Evaluator(get_system_prompt(use_tool=True))
    else:
        tool_evaluator = None

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

    # Read input file and process each row
    with open(args.response_file, mode="r", newline="") as infile:
        reader = csv.DictReader(infile)
        fields_to_keep = [
            "prompt_id",
            "capability",
            "difficulty",
            "l1_category",
            "l2_category",
            "prompt",
            "attached_file",
            "response_1",
            "response_1_human_1_rating",
            "response_1_human_1_explanation",
            "response_1_human_2_rating",
            "response_1_human_2_explanation",
            "response_2",
            "response_2_human_1_rating",
            "response_2_human_1_explanation",
            "response_2_human_2_rating",
            "response_2_human_2_explanation",
            "response_3",
            "response_3_human_1_rating",
            "response_3_human_1_explanation",
            "response_3_human_2_rating",
            "response_3_human_2_explanation",
            f"{args.response}",
        ]
        # Check if all fields_to_keep are in the original file
        missing_fields = [
            field for field in fields_to_keep if field not in reader.fieldnames
        ]
        if missing_fields:
            raise ValueError(f"Missing fields in the input file: {missing_fields}")

        # Determine fieldnames: existing fields plus the LLM-based rating and explanation
        fieldnames = [
            field for field in reader.fieldnames if field in fields_to_keep
        ] + [f"{args.response}_rating", f"{args.response}_explanation"]

        # Open the output CSV file in append mode
        with open(args.save_path, mode="a", newline="", buffering=1) as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            # Write header only if the file is new or empty
            if is_file_empty(args.save_path):
                writer.writeheader()

            # Generate evaluation for each response
            for row in tqdm(reader, desc="Processing rows"):
                if row["prompt_id"] not in processed_ids:  # Process only new rows
                    try:
                        instance_with_rating = get_judge(
                            row,
                            args.response,
                            evaluator,
                            tool_evaluator,
                            args.evaluator,
                            print_evaluation=True,
                        )
                        instance_with_rating = {
                            field: instance_with_rating[field] for field in fieldnames
                        }
                        writer.writerow(instance_with_rating)
                        processed_ids.add(row["prompt_id"])
                    except Exception as e:
                        print(
                            f"Error processing row with prompt_id {row['prompt_id']}: {e}"
                        )
                        break

    # Print average scores for each capability
    print_scores(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model responses using point deduction-based prompting with reference examples"
    )

    parser.add_argument("--response", type=str, help="Model response identifier")
    parser.add_argument(
        "--response_file", type=str, help="File path to the generated response file"
    )
    parser.add_argument(
        "--save_path", type=str, help="File path to save the evaluation results"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="gpt",
        choices=["gpt", "claude"],
        help="LLM used as evaluator",
    )
    parser.add_argument(
        "--enable_code_interpreter",
        action="store_true",
        help="Enable code interpreter for LLM-as-a-Judge (higher cost, almost same correlation, not recommended)",
    )
    args = parser.parse_args()

    main(args)
