import unittest
from unittest.mock import patch
import ollama

from core.utils import OllamaService


class OllamaServiceTests(unittest.TestCase):
    def setUp(self):
        self.model = "tinyllama:latest"
        ollama.pull(self.model)
        self.service = OllamaService(self.model)

    def test_embed(self):
        text = "Hello, I am a test"
        result = self.service.embed(text)
        self.assertIsNotNone(result)

    def test_embedChucks(self):
        text = ["Hello, I am a test", "I am a test"]
        result = self.service.embedChucks(text)
        self.assertIsNotNone(result)

    def test_check(self):
        text = "Hello, I am a test"
        result = self.service.check(text)
        self.assertTrue(result)

    def test_complete(self):
        text = "Hello, I am a test"
        max_output = 1
        result = self.service.complete(text, max_output)
        self.assertIsNotNone(result)
if __name__ == '__main__':
    unittest.main()
