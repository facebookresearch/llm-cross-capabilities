# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import requests
import tiktoken


def token_count(text):
    """
    Counts the number of tokens in the given text using tiktoken.

    Parameters:
        text (str): The text to tokenize.

    Returns:
        int: The number of tokens.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def truncate(text, model="gpt", chunk_size=512):
    """
    Truncates the text to fit within the maximum token limit of the specified model.

    Parameters:
        text (str): The text to truncate.
        model (str): The model type ('gpt', 'claude', 'gemini', 'reka', etc.).
        chunk_size (int): The size of text chunks to remove during truncation.

    Returns:
        str: The truncated text.
    """
    if model in ["claude", "gemini"]:
        max_length = 163840
    elif model in ["gpt", "llama"]:
        max_length = 102400
    elif model in ["reka"]:
        # Avoid "The read operation timed out"
        max_length = 81920
    else:
        raise ValueError(f"{model} is not supported!")
    cnt = 0

    # Truncate the text until it fits within the max token limit
    while token_count(text) > max_length:
        text = " ".join(text.split(" ")[: max_length - cnt])
        cnt += chunk_size

    return text


def get_user_prompt(instance, response, cur_principle, idx, file_url=None, model="gpt"):
    """
    Constructs the user prompt for analyzing model responses and updating principles.

    Parameters:
        instance (dict): The current instance, including the attached file, user prompt, and reference examples.
        response (str): The key for the model's response in the instance.
        cur_principle (str): The current set of principles.
        idx (int): The index of the current instance.
        file_url (str, optional): URL of the attached file, if any.
        model (str): The model type for token count and truncation.

    Returns:
        str: The constructed input prompt.
    """
    user_prompt = f"""
You are an AI expert tasked with analyzing common mistakes in model responses and creating a comprehensive set of principles to improve the **{instance['capability']}** of the model. We will work step-by-step to build this guideline. Specifically, for each iteration, I will provide you with one instance, and you need to update the current principles accordingly. There are 100 instances in total, and the principles should be completed after reviewing all instances.

For each instance, you have the following information:
- User Prompt
- Model Response
- Evaluation of the Model Response
- Current Principles

### Instance {idx}
"""
    # Include attached text for long context
    if file_url:
        attached_text = requests.get(file_url).text.strip()
        attached_text = truncate(attached_text, model)
        user_prompt += f"""
**Attached Text:**
{attached_text}
"""
    user_prompt += f"""
**User Prompt:**
{instance['prompt']}

**Model Response:**
{instance[response]}

**Evaluation:**
{instance[f'{response}_explanation']}

For each iteration, choose **ONE of the following actions**:

1. **ADD**
    - Introduce a new principle that isn't currently listed.
2. **REPLACE**
    - Replace a less significant principle with a new one.
    - Clearly specify which principle is being replaced.
3. **REVISE**
    - Enhance the principles by making them more detailed and specific.
4. **KEEP**
    - If the current instance is already covered by existing priciples, leave the guideline unchanged.

**Current Principles:**
{cur_principle}

**Output Format:**

## Summary

- Summarize any major issues with the present response.
- Provide specific, actionable steps to prevent these errors, if any.
- Based on your summary and the current principles, decide which action (ADD, REPLACE, REVISE, or KEEP) should be taken for the current instance.

## Principles for Prompts related to {instance['capability']}

### Principle 1: Title [Use the title to specify the context in which this principle should be applied, such as "For Legal Reasoning" or "For Mathematical Reasoning."]
- Include up to three key points.
- Each point should be directly applicable to the model's generation process without requiring additional training or resources.
- Each point must be extremely specific to allow for direct execution.
    - For example, instead of saying "use a structured markdown format," clearly define the exact format for each step, including the structure for the beginning, middle, and end.
    - Instead of advising to "avoid vague terms," provide a specific list of terms to be avoided.
    - Rather than generally suggesting "avoid errors in math calculations" or "double-check," outline concrete steps to prevent such errors.

...

[END of Principles]

**Requirements:**
- Follow the output format exactly, including "[END of Principles]" at the end with no remarks after it.
- Include up to 10 distinct principles in the report. If there are already 10 principles, "ADD" is not allowed.
- You may reorder the principles as necessary: Place important, typical, and representative principles at the front, while less important ones can be moved toward the back.
- **Ensure that each suggestion in the principles is detailed and actionable, rather than being a general description.**
"""
    # Return the assembled input prompt
    return user_prompt.strip()
