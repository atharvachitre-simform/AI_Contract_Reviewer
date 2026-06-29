import sys

sys.path.insert(0, ".")
import json
import logging

from src.prompts.clause_extractor_prompt import build_clause_extractor_prompt
from src.services.azure_clients import AzureOpenAIWrapper

logging.basicConfig(level=logging.INFO)

# Retrieve contract text
with open("src/prompts/system_context.py", "r") as f:
    sample_text = f.read()

# Let's mock a contract or load the contract if we can find one.
# Let's find recent uploads.
import os

pdf_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pdf") or file.endswith(".txt"):
            pdf_files.append(os.path.join(root, file))

print("Found files:", pdf_files[:5])
