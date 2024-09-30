# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import pandas as pd


def calculate_correlations(df):
    """
    Calculates Pearson, Spearman, and Kendall correlations between LLM ratings and average human ratings.

    Parameters:
        df (DataFrame): The DataFrame containing the evaluation results.

    Returns:
        dict: A dictionary containing correlation coefficients for each method and response.
    """
    correlation_results = {"pearson": {}, "spearman": {}, "kendall": {}}

    for i in range(1, 4):
        # Compute the average human rating for each response
        df[f"average_response_{i}_rating"] = df[
            [f"response_{i}_human_1_rating", f"response_{i}_human_2_rating"]
        ].mean(axis=1)

        # Pearson correlation
        pearson_corr = df[f"average_response_{i}_rating"].corr(
            df[f"response_{i}_llm_rating"], method="pearson"
        )
        correlation_results["pearson"][f"response_{i}"] = pearson_corr

        # Spearman correlation
        spearman_corr = df[f"average_response_{i}_rating"].corr(
            df[f"response_{i}_llm_rating"], method="spearman"
        )
        correlation_results["spearman"][f"response_{i}"] = spearman_corr

        # Kendall correlation
        kendall_corr = df[f"average_response_{i}_rating"].corr(
            df[f"response_{i}_llm_rating"], method="kendall"
        )
        correlation_results["kendall"][f"response_{i}"] = kendall_corr

    # Overall correlations across all responses
    all_average_ratings = pd.concat(
        [df[f"average_response_{i}_rating"] for i in range(1, 4)]
    )
    all_llm_ratings = pd.concat([df[f"response_{i}_llm_rating"] for i in range(1, 4)])

    # Overall Pearson correlation
    overall_pearson_corr = all_average_ratings.corr(all_llm_ratings, method="pearson")
    correlation_results["pearson"]["overall"] = overall_pearson_corr

    # Overall Spearman correlation
    overall_spearman_corr = all_average_ratings.corr(all_llm_ratings, method="spearman")
    correlation_results["spearman"]["overall"] = overall_spearman_corr

    # Overall Kendall correlation
    overall_kendall_corr = all_average_ratings.corr(all_llm_ratings, method="kendall")
    correlation_results["kendall"]["overall"] = overall_kendall_corr

    return correlation_results


def calculate_capability_correlations(df):
    """
    Calculates Pearson correlations for each capability between LLM ratings and average human ratings.

    Parameters:
        df (DataFrame): The DataFrame containing the evaluation results.

    Returns:
        dict: A dictionary with capabilities as keys and Pearson correlation coefficients as values.
    """
    capability_correlations = {}
    capabilities = df["capability"].unique()

    for capability in capabilities:
        subset = df[df["capability"] == capability]
        # Combine average ratings and LLM ratings for all responses within the capability
        all_average_ratings = pd.concat(
            [subset[f"average_response_{i}_rating"] for i in range(1, 4)]
        )
        all_llm_ratings = pd.concat(
            [subset[f"response_{i}_llm_rating"] for i in range(1, 4)]
        )

        # Calculate Pearson correlation for the current capability
        pearson_corr = all_average_ratings.corr(all_llm_ratings, method="pearson")
        capability_correlations[capability] = pearson_corr

    return capability_correlations


def main(args):
    # Load the data into a DataFrame
    df = pd.read_csv(args.correlation_file)

    # Calculate correlations between LLM ratings and average human ratings
    correlation_results = calculate_correlations(df)
    print("\nCorrelations:")
    for method, correlations in correlation_results.items():
        print(f"\n{method.capitalize()} Correlations:")
        for response, corr in correlations.items():
            print(f"{response} Correlation: {corr:.3f}")

    # Calculate Pearson correlations for each capability
    capability_correlations = calculate_capability_correlations(df)

    print("\nPearson Correlations by Capability:")
    for capability, corr in capability_correlations.items():
        print(f"{capability:<30} {corr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate correlations between LLM and human ratings."
    )

    parser.add_argument(
        "--correlation_file",
        type=str,
        required=True,
        help="File path to the LLM ratings for the reference responses",
    )
    args = parser.parse_args()

    main(args)
