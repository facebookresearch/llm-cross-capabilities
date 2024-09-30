# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import time
import logging
import requests
import pandas as pd

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class Gemini:
    def __init__(self, model_version, max_tokens=8192):
        """
        Initialize the Gemini client with the specified model version and maximum tokens.
        """
        self.api_key = os.environ["GOOGLE_API_KEY"]
        # Configure the API key for the generative AI client
        genai.configure(api_key=self.api_key)
        self.model_version = model_version

        # Initialize the generative model with the specified configuration
        self.model = genai.GenerativeModel(
            # Models used in our paper: 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.5-pro-exp-0801'
            self.model_version,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
            ),
        )

        # Define safety settings to handle different types of harmful content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def upload_file(self, file_url):
        """
        Upload an input file to Google Cloud for use with the generative model.

        Parameters:
            file_url (str): The URL of the file to upload.

        Returns:
            sample_file: The uploaded file object.
        """
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file: {response.status_code}")

        # Extract the file name from the URL
        file_name = file_url.split("/")[-1]
        file_path = f"downloaded_{file_name}"

        # Save the file locally
        with open(file_path, "wb") as file:
            file.write(response.content)

        # Upload the file to Google Cloud
        sample_file = genai.upload_file(path=file_path, display_name=file_name)

        # Remove the local temporary file
        os.remove(file_path)

        return sample_file

    def get_response(
        self,
        user_prompt,
        file_url=None,
        use_tool=False,
        max_retries=5,
    ):
        """
        Get a response from the Gemini model, with optional file attachment and tool usage.

        Parameters:
            user_prompt (str): The user's prompt.
            file_url (str, optional): URL of a file to send to the message.
            use_tool (bool): Whether to use code execution tools.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            str: The response from the Gemini model.
        """
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Upload the file if a file URL is provided
                cur_file = self.upload_file(file_url) if file_url is not None else None
                # Prepare the message for the model
                message = [user_prompt, cur_file] if cur_file else user_prompt
                if use_tool:
                    # Generate content using code execution tools
                    response = self.model.generate_content(
                        message,
                        safety_settings=self.safety_settings,
                        tools="code_execution",
                    )
                else:
                    # Generate content without using tools
                    response = self.model.generate_content(
                        message,
                        safety_settings=self.safety_settings,
                    )

                # If the response is not empty, return it
                if response.text.strip() != "":
                    return response.text.strip()

            except Exception as e:
                logging.error(f"Error: {e}.")

            # Increment retry count and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(min(30, 0.5 * (2**retry_interval_exp)))
                retry_interval_exp += 1

        return f"Failed to get the response after {max_retries} retries."
