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


def get_tinyurl(long_url):
    """
    Shortens a long URL using the TinyURL API.

    Parameters:
        long_url (str): The original long URL.

    Returns:
        str: The shortened URL if successful, otherwise the original URL.
    """
    tinyurl_api = f"http://tinyurl.com/api-create.php?url={long_url}"
    response = requests.get(tinyurl_api)
    if response.status_code == 200:
        return response.text
    else:
        return long_url


def token_count(text):
    """
    Counts the number of tokens in the given text using the tiktoken library.

    Parameters:
        text (str): The text to tokenize.

    Returns:
        int: The number of tokens in the text.
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
        max_length = 100000
    elif model in ["reka"]:
        # Avoid "The read operation timed out" for Reka models
        max_length = 81920
    else:
        raise ValueError(f"{model} is not supported!")
    cnt = 0

    # Truncate the text until it fits within the max token limit
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

    # Reduce image dimensions iteratively until size is acceptable
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

            # Re-encode the resized image
            final_output = io.BytesIO()
            img.save(final_output, format=img.format)
            image_data = base64.b64encode(final_output.getvalue()).decode("utf-8")
            media_type = f"image/{media_type.lower()}"

        return media_type, image_data
    else:
        raise Exception(f"Failed to retrieve image from URL: {url}")


def get_system_prompt(enable_code_interpreter=False):
    """
    Generates a system prompt based on whether the code interpreter is enabled.

    Parameters:
        enable_code_interpreter (bool): Flag to enable code interpreter features.

    Returns:
        str or None: The system prompt if enabled, otherwise None.
    """
    if enable_code_interpreter:
        system_prompt = "You are a helpful AI assistant to generate responses for the user prompt. You can write and run code to answer questions or load attached files when necessary."
    else:
        system_prompt = None
    return system_prompt


def get_user_prompt(prompt, file_url=None, model="gpt"):
    """
    Constructs the user prompt, optionally including text from an attached file.

    Parameters:
        prompt (str): The user's original prompt.
        file_url (str, optional): URL of the file to include in the prompt.
        model (str): The model type for token count and truncation.

    Returns:
        str: The constructed user prompt.
    """
    user_prompt = ""

    # Include attached text if a file URL is provided (long context)
    if file_url:
        attached_text = requests.get(file_url).text.strip()
        attached_text = truncate(attached_text, model)
        user_prompt += f"""
Attached:
{attached_text}
"""

    user_prompt += f"""
User Prompt:
{prompt}
"""
    return user_prompt.strip()
