from typing import Any, Type

import requests
from pydantic import BaseModel, Field
from requests.exceptions import RequestException


class APIError(Exception):
    """Custom exception for API related errors"""

    pass


class APIHandler:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url

    def _convert_schema(self, schema: Type[BaseModel]) -> dict[str, Any]:
        """Convert Pydantic model to JSON schema format."""
        return schema.model_json_schema()

    def prepare_request(self, prompt: str, schema: Type[BaseModel]) -> dict[str, Any]:
        """
        Prepare the request payload with proper schema conversion.

        Args:
            prompt: The input text prompt
            schema: A Pydantic model class

        Returns:
            Dict containing the formatted request payload
        """

        return {"prompt": prompt, "schema": self._convert_schema(schema)}

    def generate(self, prompt: str, schema: Type[BaseModel]) -> Any:
        """
        Send generation request to the API and return the response.

        Args:
            prompt: The input text prompt
            schema: A Pydantic model class defining the output structure

        Returns:
            The API response parsed according to the provided schema

        Raises:
            APIError: If the API request fails or returns an error
        """
        try:
            payload = self.prepare_request(prompt, schema)
            response = requests.post(f"{self.base_url}/generate", json=payload)

            if response.status_code != 200:
                raise APIError(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
            return response.json()
            # result = response.json()
            # return schema.model_validate(result)

        except RequestException as e:
            raise APIError(f"Failed to connect to API: {str(e)}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")


# Example usage:
if __name__ == "__main__":

    class Person(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age in years")
        sex: str = Field(description="The person's biological sex")

    handler = APIHandler()
    try:
        result = handler.generate("generate a person for me", Person)
        print(result)
        # print("Generated Person:", result.model_dump())
    except APIError as e:
        print(f"Error: {e}")
