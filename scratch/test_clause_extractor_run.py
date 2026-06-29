import sys

sys.path.insert(0, ".")
import json
import logging

from src import config
from src.agents.clause_extractor import ClauseExtractorAgent
from src.prompts.clause_extractor_prompt import build_clause_extractor_prompt
from src.services.azure_clients import AzureOpenAIWrapper

logging.basicConfig(level=logging.INFO)

# Load raw contract
with open("scratch/contract_46ce978f8a9180f4.txt", "r") as f:
    contract_text = f.read()

# Initialize LLM client
api_key = config.os.getenv("AZURE_OPENAI_API_KEY")
endpoint = config.os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = config.os.getenv("AZURE_OPENAI_DEPLOYMENT_CLAUSE_EXTRACTOR", "GPT-4o")

client = AzureOpenAIWrapper(endpoint=endpoint, api_key=api_key, deployment_name=deployment)

print("Running clause extractor agent...")
agent = ClauseExtractorAgent(llm_client=client)
output = agent.extract(contract_text)

print("-" * 60)
print(f"Extraction completed!")
print(f"Is complete: {output.is_extraction_complete}")
print(f"Clause count: {len(output.clauses)}")
print(f"Highest clause number: {output.highest_clause_number}")
print(f"Notes: {output.extraction_completeness_notes}")

# Save the LLM raw response if available
last_resp = getattr(client, "_last_response", None)
if last_resp:
    print("Finish reason:", last_resp.choices[0].finish_reason)
    print("Output length:", len(last_resp.choices[0].message.content))
    with open("scratch/raw_clause_extractor_response.txt", "w") as out:
        out.write(last_resp.choices[0].message.content)
