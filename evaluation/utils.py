# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import io
import base64
import httpx
import requests
import tiktoken
from PIL import Image


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


def truncate(text, evaluator="gpt", chunk_size=512):
    """
    Truncates the text to fit within the maximum token limit of the evaluator.

    Parameters:
        text (str): The text to truncate.
        evaluator (str): The evaluator model ('gpt' or 'claude').
        chunk_size (int): The size of text chunks to remove during truncation.

    Returns:
        str: The truncated text.
    """
    if evaluator == "claude":
        max_length = 160000
    else:
        max_length = 102400
    cnt = 0
    while token_count(text) > max_length:
        text = " ".join(text.split(" ")[: max_length - cnt])
        cnt += chunk_size
    return text


def resize_image(image, target_size, format):
    """
    Resizes an image to ensure its base64-encoded size does not exceed the target size.

    Parameters:
        image (PIL.Image.Image): The image to resize.
        target_size (int): The maximum allowed size in bytes.
        format (str): The image format ('PNG', 'JPEG', etc.).

    Returns:
        PIL.Image.Image: The resized image.
    """
    width, height = image.size
    aspect_ratio = width / height

    # Reduce dimensions iteratively until the size is acceptable
    while True:
        new_width = int(width * 0.9)
        new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        image.save(output, format=format)
        b64_image_data = base64.b64encode(output.getvalue()).decode("utf-8")
        b64_image_size = len(b64_image_data)

        if b64_image_size <= target_size:
            break

        width, height = image.size

    output.seek(0)
    return Image.open(output)


def get_image_from_url(url):
    """
    Downloads an image from a URL, resizes it if necessary, and encodes it in base64.

    Parameters:
        url (str): The URL of the image.

    Returns:
        tuple: A tuple containing the media type and base64-encoded image data.
    """
    response = httpx.get(url)
    if response.status_code == 200:
        image_file = io.BytesIO(response.content)
        with Image.open(image_file) as img:
            media_type = img.format
            if media_type not in ["PNG", "JPEG", "WEBP", "GIF"]:
                raise ValueError(f"The image format {media_type} is not supported.")

            # Encode the initial image to base64
            initial_output = io.BytesIO()
            img.save(initial_output, format=img.format)
            initial_b64_image_data = base64.b64encode(initial_output.getvalue()).decode(
                "utf-8"
            )
            initial_b64_image_size = len(initial_b64_image_data)
            max_size = 5 * 1024 * 1024  # 5MB limitation for Claude API

            # Resize the image if it exceeds the max size
            if initial_b64_image_size > max_size:
                img = resize_image(img, max_size, img.format)

            # Re-encode the image after resizing
            final_output = io.BytesIO()
            img.save(final_output, format=img.format)
            image_data = base64.b64encode(final_output.getvalue()).decode("utf-8")
            media_type = f"image/{media_type.lower()}"

        return media_type, image_data
    else:
        raise Exception(f"Failed to retrieve image from URL: {url}")


def get_system_prompt(use_tool=False):
    """
    Generates the system prompt for the AI evaluator, optionally including instructions for tool use.

    Parameters:
        use_tool (bool): Whether to include instructions about using tools.

    Returns:
        str: The system prompt.
    """
    system_prompt = """
You are an expert AI evaluator tasked with assessing model responses. Rate the response using a 1-5 Likert scale according to the following rubrics:

### Rubrics:
- **5/5 - Amazing**: The response is flawless and could hardly be improved.
- **4/5 - Pretty Good**: The response is quite good, but has room for minor improvements.
- **3/5 - Okay**: They are middle of the road responses that could be improved in several ways.
- **2/5 - Pretty Bad**: The response has major problems in helpfulness, truthfulness, or safety.
- **1/5 - Horrible**: They are terrible responses and you would caution others against using models that generate responses like this.
"""
    if use_tool:
        system_prompt += """
Note: User prompts or model responses may include attachments. To ensure a thorough evaluation, you may need to write and execute code.
"""
    return system_prompt.strip()


def get_user_prompt_deduct_wo_ref(prompt, response, instance, evaluator="gpt"):
    """
    Constructs the user prompt for evaluation without reference examples, including point deduction-based prompts.

    Parameters:
        prompt (str): The user's original prompt.
        response (str): The model's response to be evaluated.
        instance (dict): The current instance, including the attached file and user prompt.
        evaluator (str): The evaluator model ('gpt' or 'claude').

    Returns:
        str: The constructed user prompt.
    """
    # Include attached text for long context
    if instance["attached_file"] != "" and (
        "long context" in instance["capability"].lower()
    ):
        attached_text = requests.get(instance["attached_file"]).text.strip()
        attached_text = truncate(attached_text, evaluator)
        user_prompt = f"""
[Attached]:
{attached_text}
"""
    else:
        user_prompt = ""

    user_prompt += f"""
[User Prompt]:
{prompt.strip()}

Here is the model response for evaluation:

[Model Response to be Evaluated]:
{response.strip()}

Please provide your evaluation in the following format:

#### User Prompt Analysis
- Identify key requirements and objectives from the user prompt.

#### Model Response Evaluation
- Pros: List strengths and positive aspects.
- Cons: Identify weaknesses, **specifying point deductions for each**.

#### Holistic Assessment
- Consider if major strengths outweigh minor issues.
- Combine similar deductions to avoid double penalization.
- Balance deductions and positive aspects, and then explain your scoring decision.

#### Evaluation Score
Score: [X]/5

Ensure your evaluation is thorough, fair, and aligned with the point deductions. Your expertise is crucial in providing an accurate and insightful assessment.
"""
    if instance["capability"].lower() == "tool use":
        user_prompt = (
            user_prompt.strip()
            + " For user prompts requiring real-time information, remember that answers can change over time. **DO NOT rely on your own knowledge to make judgments or deductions.** Instead, primarily evaluate whether the model has conducted a web search and if the response meets user requirements."
        )
    # tool use & coding, tool use & reasoning
    elif "tool use" in instance["capability"].lower():
        user_prompt = (
            user_prompt.strip()
            + " A response containing an attached URL or image link indicates that the model has executed the code and provided a file or image. Failure to execute the code, or not providing a link for the produced image or file, should result in a major deduction."
        )
    return user_prompt.strip()


def get_user_prompt_with_1ref(prompt, response, instance, idx, evaluator="gpt"):
    """
    Constructs the user prompt for evaluation with one reference example.

    Parameters:
        prompt (str): The user's original prompt.
        response (str): The model's response to be evaluated.
        instance (dict): The current instance, including the attached file, user prompt, and reference examples.
        idx (int): The index of the current instance.
        evaluator (str): The evaluator model ('gpt' or 'claude').

    Returns:
        str: The constructed user prompt.
    """
    # Determine indices for the reference example
    example_1_idx = (idx % 3) + 1

    # Include attached text for long context
    if instance["attached_file"] != "" and (
        "long context" in instance["capability"].lower()
    ):
        attached_text = requests.get(instance["attached_file"]).text.strip()
        attached_text = truncate(attached_text, evaluator)
        user_prompt = f"""
[Attached]:
{attached_text}
"""
    else:
        user_prompt = ""

    user_prompt += f"""
[User Prompt]:
{prompt.strip()}

To calibrate your evaluation, consider these reference examples:

[Reference Example]:
Model Response: {instance[f'response_{example_1_idx}']}
Rating 1: {instance[f'response_{example_1_idx}_human_1_rating']}/5 | Explanation 1: {instance[f'response_{example_1_idx}_human_1_explanation'].strip()}
Rating 2: {instance[f'response_{example_1_idx}_human_2_rating']}/5 | Explanation 2: {instance[f'response_{example_1_idx}_human_2_explanation'].strip()}

**Use these examples as benchmarks for your evaluation scale and scoring consistency.** Here is the model response for evaluation:

[Model Response to be Evaluated]:
{response.strip()}

Please provide your evaluation in the following format:

#### User Prompt Analysis
- Identify key requirements and objectives from the user prompt.

#### Reference Examples Insights
- Summarize scoring patterns and typical point deductions.
- Include how many points should be deducted for each issue.

#### Model Response Evaluation
- Pros: List strengths and positive aspects.
- Cons: Identify weaknesses, **specifying point deductions for each**.

#### Holistic Assessment
- Consider if major strengths outweigh minor issues.
- Combine similar deductions to avoid double penalization.
- Balance deductions and positive aspects, and then explain your scoring decision.

#### Evaluation Score
Score: [X]/5

Ensure your evaluation is thorough, fair, and aligned with the reference examples. Your expertise is crucial in providing an accurate and insightful assessment.
"""
    if instance["capability"].lower() == "tool use":
        user_prompt = (
            user_prompt.strip()
            + " The model responses in reference examples were generated in June-July 2024. For user prompts requiring real-time information, remember that answers can change over time. **DO NOT rely on your own knowledge to make judgments or deductions.** Instead, primarily evaluate whether the model has conducted a web search and if the response meets user requirements."
        )
    # tool use & coding, tool use & reasoning
    elif "tool use" in instance["capability"].lower():
        user_prompt = (
            user_prompt.strip()
            + " A response containing an attached URL or image link indicates that the model has executed the code and provided a file or image. Failure to execute the code, or not providing a link for the produced image or file, should result in a major deduction."
        )
    return user_prompt.strip()


def get_user_prompt_with_2ref(prompt, response, instance, idx, evaluator="gpt"):
    """
    Constructs the user prompt for evaluation with two reference examples.

    Parameters:
        prompt (str): The user's original prompt.
        response (str): The model's response to be evaluated.
        instance (dict): The current instance, including the attached file, user prompt, and reference examples.
        idx (int): The index of the current instance.
        evaluator (str): The evaluator model ('gpt' or 'claude').

    Returns:
        str: The constructed user prompt including reference examples.
    """
    # Determine indices for the two reference examples
    example_1_idx = (idx % 3) + 1
    example_2_idx = (example_1_idx % 3) + 1

    # Include attached text for long context
    if instance["attached_file"] != "" and (
        "long context" in instance["capability"].lower()
    ):
        attached_text = requests.get(instance["attached_file"]).text.strip()
        attached_text = truncate(attached_text, evaluator)
        user_prompt = f"""
[Attached]:
{attached_text}
"""
    else:
        user_prompt = ""

    # Build the user prompt with user input and reference examples
    user_prompt += f"""
[User Prompt]:
{prompt.strip()}

To calibrate your evaluation, consider these reference examples:

[Reference Example 1]:
Model Response: {instance[f'response_{example_1_idx}']}
Rating 1: {instance[f'response_{example_1_idx}_human_1_rating']}/5 | Explanation 1: {instance[f'response_{example_1_idx}_human_1_explanation'].strip()}
Rating 2: {instance[f'response_{example_1_idx}_human_2_rating']}/5 | Explanation 2: {instance[f'response_{example_1_idx}_human_2_explanation'].strip()}

[Reference Example 2]:
Model Response: {instance[f'response_{example_2_idx}']}
Rating 1: {instance[f'response_{example_2_idx}_human_1_rating']}/5 | Explanation 1: {instance[f'response_{example_2_idx}_human_1_explanation'].strip()}
Rating 2: {instance[f'response_{example_2_idx}_human_2_rating']}/5 | Explanation 2: {instance[f'response_{example_2_idx}_human_2_explanation'].strip()}

**Use these examples as benchmarks for your evaluation scale and scoring consistency.** Here is the model response for evaluation:

[Model Response to be Evaluated]:
{response.strip()}

Please provide your evaluation in the following format:

#### User Prompt Analysis
- Identify key requirements and objectives from the user prompt.

#### Reference Examples Insights
- Summarize scoring patterns and typical point deductions.
- Include how many points should be deducted for each issue.

#### Model Response Evaluation
- Pros: List strengths and positive aspects.
- Cons: Identify weaknesses, **specifying point deductions for each**.

#### Holistic Assessment
- Consider if major strengths outweigh minor issues.
- Combine similar deductions to avoid double penalization.
- Balance deductions and positive aspects, and then explain your scoring decision.

#### Evaluation Score
Score: [X]/5

Ensure your evaluation is thorough, fair, and aligned with the reference examples. Your expertise is crucial in providing an accurate and insightful assessment.
"""
    if instance["capability"].lower() == "tool use":
        user_prompt = (
            user_prompt.strip()
            + " The model responses in reference examples were generated in June-July 2024. For user prompts requiring real-time information, remember that answers can change over time. **DO NOT rely on your own knowledge to make judgments or deductions.** Instead, primarily evaluate whether the model has conducted a web search and if the response meets user requirements."
        )
    # tool use & coding, tool use & reasoning
    elif "tool use" in instance["capability"].lower():
        user_prompt = (
            user_prompt.strip()
            + " A response containing an attached URL or image link indicates that the model has executed the code and provided a file or image. Failure to execute the code, or not providing a link for the produced image or file, should result in a major deduction."
        )
    return user_prompt.strip()


def get_user_prompt_with_3ref(prompt, response, instance, evaluator="gpt"):
    """
    Constructs the user prompt for evaluation with three reference examples.

    Parameters:
        prompt (str): The user's original prompt.
        response (str): The model's response to be evaluated.
        instance (dict): The current instance, including the attached file, user prompt, and reference examples.
        evaluator (str): The evaluator model ('gpt' or 'claude').

    Returns:
        str: The constructed user prompt including all reference examples.
    """
    # Include attached text for long context
    if instance["attached_file"] != "" and (
        "long context" in instance["capability"].lower()
    ):
        attached_text = requests.get(instance["attached_file"]).text.strip()
        attached_text = truncate(attached_text, evaluator)
        user_prompt = f"""
[Attached]:
{attached_text}
"""
    else:
        user_prompt = ""

    # Build the user prompt with user input and three reference examples
    user_prompt += f"""
[User Prompt]:
{prompt.strip()}

To calibrate your evaluation, consider these reference examples:

[Reference Example 1]:
Model Response: {instance[f'response_1']}
Rating 1: {instance[f'response_1_human_1_rating']}/5 | Explanation 1: {instance[f'response_1_human_1_explanation'].strip()}
Rating 2: {instance[f'response_1_human_2_rating']}/5 | Explanation 2: {instance[f'response_1_human_2_explanation'].strip()}

[Reference Example 2]:
Model Response: {instance[f'response_2']}
Rating 1: {instance[f'response_2_human_1_rating']}/5 | Explanation 1: {instance[f'response_2_human_1_explanation'].strip()}
Rating 2: {instance[f'response_2_human_2_rating']}/5 | Explanation 2: {instance[f'response_2_human_2_explanation'].strip()}

[Reference Example 3]:
Model Response: {instance[f'response_3']}
Rating 1: {instance[f'response_3_human_1_rating']}/5 | Explanation 1: {instance[f'response_3_human_1_explanation'].strip()}
Rating 2: {instance[f'response_3_human_2_rating']}/5 | Explanation 2: {instance[f'response_3_human_2_explanation'].strip()}

**Use these examples as benchmarks for your evaluation scale and scoring consistency.** Here is the model response for evaluation:

[Model Response to be Evaluated]:
{response.strip()}

Please provide your evaluation in the following format:

#### User Prompt Analysis
- Identify key requirements and objectives from the user prompt.

#### Reference Examples Insights
- Summarize scoring patterns and typical point deductions.
- Include how many points should be deducted for each issue.

#### Model Response Evaluation
- Pros: List strengths and positive aspects.
- Cons: Identify weaknesses, **specifying point deductions for each**.

#### Holistic Assessment
- Consider if major strengths outweigh minor issues.
- Combine similar deductions to avoid double penalization.
- Balance deductions and positive aspects, and then explain your scoring decision.

#### Evaluation Score
Score: [X]/5

Ensure your evaluation is thorough, fair, and aligned with the reference examples. Your expertise is crucial in providing an accurate and insightful assessment.
"""
    if instance["capability"].lower() == "tool use":
        user_prompt = (
            user_prompt.strip()
            + " The model responses in reference examples were generated in June-July 2024. For user prompts requiring real-time information, remember that answers can change over time. **DO NOT rely on your own knowledge to make judgments or deductions.** Instead, primarily evaluate whether the model has conducted a web search and if the response meets user requirements."
        )
    # tool use & coding, tool use & reasoning
    elif "tool use" in instance["capability"].lower():
        user_prompt = (
            user_prompt.strip()
            + " A response containing an attached URL or image link indicates that the model has executed the code and provided a file or image. Failure to execute the code, or not providing a link for the produced image or file, should result in a major deduction."
        )
    return user_prompt.strip()
