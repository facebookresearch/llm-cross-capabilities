# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import time
import logging

import anthropic

from .utils import get_image_from_url


class Claude:
    def __init__(self, model_version):
        """
        Initialize the Claude client with the specified model version.
        """
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_version = model_version

    def get_response(
        self,
        user_prompt,
        system_prompt,
        file_url=None,
        max_tokens=4096,
        max_retries=5,
        **kwargs,
    ):
        """
        Get a response from the Claude model, with optional file attachment.

        Parameters:
            user_prompt (str): The user's prompt.
            system_prompt (str): The system prompt to guide the assistant's behavior.
            file_url (str, optional): URL of a file to attach to the message.
            max_tokens (int): Maximum number of tokens for the response.
            max_retries (int): Maximum number of retries in case of failure.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            str: The response from the Claude model.
        """
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
                if file_url:
                    # Retrieve the image data and media type from the URL
                    media_type, image_data = get_image_from_url(file_url)

                    # Include the image in the message content
                    messages[-1]["content"] = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ]

                # Prepare parameters for the API call
                params = {
                    "model": self.model_version,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                if system_prompt is not None:
                    # Include the system prompt if provided
                    params["system"] = system_prompt

                # Call the Anthropic API to get the response
                response = self.client.messages.create(**params)

                # Extract text from the response content blocks
                text_content = ""
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                if text_content.strip() != "":
                    return text_content.strip()
            except Exception as e:
                logging.error(f"Error: {e}.")

            # Increment retry count and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(min(30, 0.5 * (2**retry_interval_exp)))
                retry_interval_exp += 1
        return f"Failed to get the response after {max_retries} retries."
