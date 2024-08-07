import json
from abc import ABC, abstractmethod

import ollama
from openai import BadRequestError
from openai.lib.azure import AzureOpenAI


class KeyLoader(ABC):
    @abstractmethod
    def key(self) -> str:
        pass


class FileKeyLoader(KeyLoader):
    def __init__(self, filename: str):
        self.filename = filename

    def key(self) -> str:
        # open file and return key
        with open(self.filename, 'r') as f:
            return f.read()


class LlmService:

    def embed(self, text: str): pass

    def embedChucks(self, text: list[str]): pass

    def check(self, text: str) -> bool: pass

    def complete(self, text: str, max_output: int) -> str: pass

    @staticmethod
    def from_file(where: str, filename: str):
        with open(where + "/" + filename, 'r') as f:
            data = json.load(f)
            if data["type"] == "OpenAi":
                loader = FileKeyLoader(data["keyfile"])
                return OpenAiService(loader, data["endpoint"], data["deployment"], data["version"], data["model"])
            elif data["type"] == "Ollama":
                return OllamaService(data["model"])

class OpenAiService(LlmService):
    def __init__(self, api_loader: KeyLoader, endpoint: str, deployment: str, version: str, model: str):
        self.key = api_loader.key()
        self.model = model
        self.service = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=self.key,
            api_version=version
        )

    def embed(self, text: str):
        return self.service.embeddings.create(model=self.model, input=text).data[0].embedding

    def embedChucks(self, text: list[str]):
        all = self.service.embeddings.create(model=self.model, input=text).data
        return [a.embedding for a in all]

    def check(self, text: str) -> (bool, object):
        try:
            self.service.completions.create(
                model=self.model,
                prompt=text,
                max_tokens=1
            )
            return True, {}
        except BadRequestError as e:
            return False, e

    def complete(self, text: str, max_output: int) -> str:
        result = None
        while result is None:
            result = self.service.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                max_tokens=max_output,
                temperature=1,
                top_p=0.5,
                frequency_penalty=0.0,
                presence_penalty=0,
                stop=None
            ).choices[0].message.content
        return result
    # static factory method to load from a file
    @staticmethod
    def from_file(where: str, filename: str):
        # open as json
        with open(where + "/" + filename, 'r') as f:
            data = json.load(f)
            loader = FileKeyLoader(data["keyfile"])
            return OpenAiService(loader, data["endpoint"], data["deployment"], data["version"], data["model"])

class OllamaService(LlmService):
    def __init__(self, model: str):
        self.model = model

    def embed(self, text: str):
        return ollama.embeddings(model=self.model, prompt=text)['embedding']

    def embedChucks(self, text: list[str]):
        return [self.embed(t) for t in text]

    def check(self, text: str) -> (bool, object):
        return True, {}

    def complete(self, text: str, max_output: int) -> str:
        return ollama.generate(model=self.model, prompt=text, options={"eval_count": max_output})["response"]

    @staticmethod
    def from_file(where: str, filename: str):
        with open(where + "/" + filename, 'r') as f:
            data = json.load(f)
            return OllamaService(data["model"])
