# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import time
import logging

import openai
from openai import OpenAI


class GPT:
    def __init__(self, model_version="gpt-4o-2024-05-13", system_prompt=None):
        """
        Initialize the GPT client with the specified model version and optional system prompt.
        """
        self.api_key = os.environ["OPENAI_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.model_version = model_version
        self.system_prompt = system_prompt

    def get_response(
        self,
        user_prompt,
        image_url=None,
        max_tokens=4096,
        max_retries=5,
        **kwargs,
    ):
        """
        Get a response from the GPT model, with optional file attachment.

        Parameters:
            user_prompt (str): The user's prompt.
            file_url (str, optional): URL of a file to attach to the prompt.
            max_tokens (int): Maximum number of tokens for the response.
            max_retries (int): Maximum number of retries in case of failure.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            str: The response from the GPT model.
        """
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                messages = []
                if self.system_prompt:
                    # Add the system prompt to the messages
                    messages.append({"role": "system", "content": self.system_prompt})

                # Add the user prompt
                messages.append({"role": "user", "content": user_prompt})

                if image_url:
                    # If an image URL is provided, include it in the message content
                    messages[-1]["content"] = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ]

                # Call the OpenAI API to get the response
                response = self.client.chat.completions.create(
                    model=self.model_version,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    **kwargs,
                )

                # If the response is not empty, return it
                if response.choices[0].message.content.strip() != "":
                    return response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                logging.warning(f"OpenAI rate limit error: {e}. Retry!")
            except openai.APIConnectionError as e:
                logging.warning(f"OpenAI API connection error: {e}. Retry!")
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}. Retry!")
            except Exception as e:
                logging.error(f"Unexpected error: {e}.")

            # Increment retry count and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(min(30, 0.5 * (2**retry_interval_exp)))
                retry_interval_exp += 1
        return f"Failed to get the response after {max_retries} retries."
