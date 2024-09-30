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

import anthropic
import openai
from openai import OpenAI

from utils import get_image_from_url


class GPT_Evaluator:
    def __init__(self, system_prompt):
        """
        Initialize the GPT_Evaluator with a system prompt.
        """
        self.api_key = os.environ["OPENAI_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt

    def evaluate(
        self,
        user_prompt,
        image_url=None,
        max_tokens=4096,
        temperature=0.0,
        seed=42,
        max_retries=5,
        **kwargs,
    ):
        """
        Evaluate the user's prompt using the GPT model, with optional image attachment.

        Parameters:
            user_prompt (str): The user's prompt to evaluate.
            image_url (str, optional): URL of an image to include in the evaluation.
            max_tokens (int): Maximum number of tokens for the response.
            temperature (float): Sampling temperature for the model.
            seed (int): Random seed for reproducibility.
            max_retries (int): Maximum number of retries in case of failure.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            str: The evaluation result from the GPT model.
        """
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Prepare messages with system and user prompts
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
                if image_url:
                    # Include the image in the user message content
                    messages[-1]["content"] = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ]
                # Call the OpenAI API to get the evaluation
                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-05-13",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    **kwargs,
                )
                return response.choices[0].message.content
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


class Tool_Evaluator:
    def __init__(self, system_prompt):
        """
        Initialize the Tool_Evaluator with a system prompt and set up the assistant.
        """
        self.api_key = os.environ["OPENAI_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        # Create an assistant with code interpreter tool
        self.assistant = self.client.beta.assistants.create(
            instructions=system_prompt,
            model="gpt-4o-2024-05-13",
            tools=[{"type": "code_interpreter"}],
        )

    def upload_file(self, file_url):
        """
        Upload a file to OpenAI assistant for use with code interpreter.

        Parameters:
            file_url (str): URL of the file to upload.

        Returns:
            openai.File: The uploaded file object.
        """
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file: {response.status_code}")

        # Determine the file type from the URL
        file_type = file_url.split(".")[-1].lower()
        file_name = f"downloaded_file.{file_type}"

        # Save the file locally
        with open(file_name, "wb") as file:
            file.write(response.content)

        # Upload the file to OpenAI assistant
        with open(file_name, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose="assistants")

        # Remove the temporary local file
        os.remove(file_name)

        return uploaded_file

    def evaluate(
        self,
        user_prompt,
        file_url=None,
        temperature=0.0,
    ):
        """
        Evaluate the user's prompt using the assistant with code interpreter.

        Parameters:
            user_prompt (str): The user's prompt to evaluate.
            file_url (str, optional): URL of a file to attach to the prompt.
            temperature (float): Sampling temperature for the model.

        Returns:
            str: The evaluation result from the assistant.
        """
        # Prepare the initial user message
        messages = [{"role": "user", "content": user_prompt}]
        if file_url:
            # Upload the file and include it in the attachments
            file = self.upload_file(file_url)
            messages[0]["attachments"] = [
                {"file_id": file.id, "tools": [{"type": "code_interpreter"}]}
            ]
        # Create a new thread for the conversation
        thread = self.client.beta.threads.create(
            messages=messages,
        )

        # Start a run with the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
            temperature=temperature,
        )

        # Poll the run status until it completes
        while run.status not in ["completed", "failed"]:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            time.sleep(5)
        if run.status == "completed":
            # Retrieve messages from the thread
            responses = self.client.beta.threads.messages.list(thread_id=thread.id)
            text_response = ""
            # Extract text content from the response
            for i in range(len(responses.data[0].content)):
                if responses.data[0].content[i].type == "text":
                    text_response += responses.data[0].content[i].text.value
            return text_response
        else:
            return "Failed to get the response."


class Claude_Evaluator:
    def __init__(self):
        """
        Initialize the Claude_Evaluator client.
        """
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def evaluate(
        self,
        user_prompt,
        system_prompt,
        image_url=None,
        max_tokens=4096,
        temperature=0.0,
        max_retries=5,
        **kwargs,
    ):
        """
        Evaluate the user's prompt using the Claude model, with optional image attachment.

        Parameters:
            user_prompt (str): The user's prompt to evaluate.
            system_prompt (str): The system prompt to guide the assistant's behavior.
            image_url (str, optional): URL of an image to include in the evaluation.
            max_tokens (int): Maximum number of tokens for the response.
            temperature (float): Sampling temperature for the model.
            max_retries (int): Maximum number of retries in case of failure.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            str: The evaluation result from the Claude model.
        """
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Prepare messages with user prompt
                messages = [
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
                if image_url:
                    # Retrieve and encode the image data
                    media_type, image_data = get_image_from_url(image_url)
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
                # Call the Anthropic API to get the evaluation
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )

                # Extract text from content blocks
                text_content = ""
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                return text_content.strip()
            except Exception as e:
                logging.error(f"Error: {e}.")

            # Increment retry count and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(min(30, 0.5 * (2**retry_interval_exp)))
                retry_interval_exp += 1
        return f"Failed to get the response after {max_retries} retries."
