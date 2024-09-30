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

import openai
from openai import OpenAI
from huggingface_hub import login, HfApi

from .utils import get_tinyurl


class GPT:
    def __init__(self, model_version, system_prompt=None):
        """
        Initialize the GPT client with the specified model version and optional system prompt.
        """
        # You need to use your own API key and comply with OpenAI terms of use.
        self.api_key = os.environ["OPENAI_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.model_version = model_version
        self.system_prompt = system_prompt

    def get_response(
        self,
        user_prompt,
        file_url=None,
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

                if file_url:
                    # If a file URL is provided, include it in the message content
                    messages[-1]["content"] = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": file_url,
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


class Tool_GPT:
    def __init__(self, model_version, system_prompt):
        """
        Initialize the Tool_GPT client with the specified model version and system prompt.
        Sets up the assistant with code interpreter and logs into HuggingFace Hub.
        """
        self.api_key = os.environ["OPENAI_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.model_version = model_version
        # Create an assistant with the code interpreter tool
        self.assistant = self.client.beta.assistants.create(
            instructions=system_prompt,
            model=self.model_version,
            tools=[{"type": "code_interpreter"}],
        )

        # Log into HuggingFace Hub using the token from environment variables
        login(token=os.environ["HF_KEY"])
        self.hf_api = HfApi()
        # Set the HuggingFace username and repository name
        self.hf_name = ""  # Add your HuggingFace username here
        self.repo_name = ""  # Add your repository name here
        # Create the dataset repository on HuggingFace Hub if it doesn't exist
        self.hf_api.create_repo(
            repo_id=self.repo_name, repo_type="dataset", exist_ok=True
        )

    # upload input file for assistant api
    def upload_file(self, file_url):
        """
        Upload an input file to OpenAI assistant.

        Parameters:
            file_url (str): URL of the file to upload.

        Returns:
            openai.File: The uploaded file object.
        """
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file: {response.status_code}")

        # Determine the file extension from the URL
        file_type = file_url.split(".")[-1].lower()
        file_name = f"downloaded_file.{file_type}"

        # Save the file locally
        with open(file_name, "wb") as file:
            file.write(response.content)

        # Upload the file to OpenAI assistant
        with open(file_name, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose="assistants")

        # Remove the local temporary file
        os.remove(file_name)

        return uploaded_file

    def upload_image(self, image_id):
        """
        Upload an image generated by the assistant to HuggingFace Hub.

        Parameters:
            image_id (str): The file ID of the image in OpenAI assistant.

        Returns:
            str: The shortened URL of the uploaded image.
        """
        # Get the image data from OpenAI files
        image_data = self.client.files.content(image_id)
        image_data_bytes = image_data.read()
        image_name = f"{image_id}.png"

        # Save the image locally
        with open(image_name, "wb") as file:
            file.write(image_data_bytes)

        # Upload the image to HuggingFace Hub dataset
        self.hf_api.upload_file(
            path_or_fileobj=image_name,
            path_in_repo=image_name,
            repo_id=f"{self.hf_name}/{self.repo_name}",
            repo_type="dataset",
        )

        # Construct the URL to the uploaded image
        hf_url = f"https://huggingface.co/datasets/{self.hf_name}/{self.repo_name}/blob/main/{image_name}"

        # Shorten the URL using TinyURL
        shorten_url = get_tinyurl(hf_url)

        # Remove the local temporary image
        os.remove(image_name)

        return shorten_url

    def get_response(self, user_prompt, file_url=None, include_image=True):
        """
        Get a response from the assistant, optionally including images.

        Parameters:
            user_prompt (str): The user's prompt.
            file_url (str, optional): URL of a file to attach to the message.
            include_image (bool): Whether to include images in the response.

        Returns:
            str: The response from the assistant.
        """
        # Prepare the messages for the assistant
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
        )

        # Poll the run status until it completes
        while run.status not in ["completed", "failed"]:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            time.sleep(5)
        if run.status == "completed":
            # Get the messages from the thread
            responses = self.client.beta.threads.messages.list(thread_id=thread.id)
            text_response = ""
            # Process each content item in the response
            for i in range(len(responses.data[0].content)):
                if responses.data[0].content[i].type == "text":
                    # Append text content
                    text_response += responses.data[0].content[i].text.value
                elif (
                    include_image and responses.data[0].content[i].type == "image_file"
                ):
                    # Upload the image and include the link
                    public_url = self.upload_image(
                        responses.data[0].content[i].image_file.file_id
                    )
                    text_response += f"\n\nImage Link: {public_url}\n\n"
            return text_response.strip()
        else:
            return "Failed to get the response."
