# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import time
import logging

from reka.client import Reka
from reka import ChatMessage


class REKA:
    def __init__(self, model_version):
        """
        Initialize the REKA client with the specified model version.
        """
        self.client = Reka(api_key=os.environ["REKA_API_KEY"])

        # Models used in our paper: "reka-edge-20240208", "reka-flash-20240722", "reka-core-20240722"
        self.model = model_version

    def get_response(
        self,
        user_prompt,
        file_url=None,
        max_tokens=4096,
        max_retries=5,
    ):
        """
        Get a response from the REKA model, with optional file attachment.

        Parameters:
            user_prompt (str): The user's prompt.
            file_url (str, optional): URL of a file to attach to the message.
            max_tokens (int): Maximum number of tokens for the response.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            str: The response from the REKA model.
        """
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                if file_url:
                    # Create a chat message with an image URL and text content
                    response = self.client.chat.create(
                        messages=[
                            ChatMessage(
                                content=[
                                    {"type": "image_url", "image_url": file_url},
                                    {"type": "text", "text": user_prompt},
                                ],
                                role="user",
                            )
                        ],
                        model=self.model,
                        max_tokens=max_tokens,
                    )
                else:
                    # Create a chat message with text content only
                    response = self.client.chat.create(
                        messages=[
                            {
                                "content": user_prompt,
                                "role": "user",
                            }
                        ],
                        model=self.model,
                        max_tokens=max_tokens,
                    )

                # Check if the response is not empty
                if response.responses[0].message.content.strip() != "":
                    return response.responses[0].message.content.strip()

            except Exception as e:
                logging.error(f"Error: {e}.")

            # Increment retry count and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(min(30, 0.5 * (2**retry_interval_exp)))
                retry_interval_exp += 1

        return f"Failed to get the response after {max_retries} retries."
