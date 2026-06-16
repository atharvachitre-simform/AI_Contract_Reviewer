"""OpenAI Batch API processor for bulk contract extraction."""

import json
import logging
import os
import tempfile
from typing import Any

from .azure_clients import AzureClientFactory
from ..agents.clause_extractor import (
    split_into_extraction_units,
    classify_extraction_unit,
    contains_risk_trigger_terms,
    _parse_llm_response,
    _build_clauses_from_llm,
    _merge_metadata,
    _build_cuad_labels,
    get_page_number_for_text,
)
from ..prompts.clause_extractor_prompt import build_clause_extractor_prompt
from ..models import ClauseExtractorOutput, ContractMetadata

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Manages bulk Clause Extraction tasks via OpenAI's Batch API."""

    def __init__(self, azure_factory: AzureClientFactory | None = None):
        self.azure = azure_factory or AzureClientFactory()

    def get_raw_openai_client(self) -> Any:
        wrapper = self.azure.get_openai_client_for_agent("clause_extractor")
        if not wrapper or not wrapper.openai_client:
            raise RuntimeError("Raw OpenAI client not configured for clause extractor.")
        return wrapper.openai_client

    def compile_extraction_batch(self, contracts: list[dict[str, str]]) -> str:
        """
        Takes a list of dicts with 'contract_id' and 'contract_text'.
        Returns the path to a generated .jsonl file containing Batch API requests.
        """
        # We must use the exact deployment name needed by the Batch API format
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_CLAUSE_EXTRACTOR", "gpt-4o")
        
        tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".jsonl")
        
        try:
            for contract in contracts:
                contract_id = contract["contract_id"]
                contract_text = contract["contract_text"]
                
                # Assume general contract type for bulk mode to keep structural chunking uniform
                contract_type = "general"
                units = split_into_extraction_units(contract_text, contract_type)
                
                for idx, unit in enumerate(units):
                    # RAG / Hierarchy Filter 
                    classif, relevance_score = classify_extraction_unit(unit["text"])
                    if classif == "PURE_DEFINITION":
                        continue
                    
                    MIN_RELEVANCE_THRESHOLD = 0.3
                    if relevance_score < MIN_RELEVANCE_THRESHOLD and not contains_risk_trigger_terms(unit["text"]):
                        continue
                    
                    target_clauses = max(3, min(20, unit["token_count"] // 120))
                    prompt = build_clause_extractor_prompt(
                        unit["text"],
                        source_file=None,
                        memory_context=None,
                        reference_clauses=None,
                        section_hint=unit["section"],
                        target_clauses=target_clauses,
                        context_header=unit["context_header"]
                    )
                    
                    sep = "INSTRUCTIONS:\n"
                    if sep in prompt:
                        system_prompt, user_prompt = prompt.split(sep, 1)
                        system_prompt = system_prompt.replace("SYSTEM:", "").strip()
                        user_prompt = sep + user_prompt
                    else:
                        system_prompt = "You are a contract analysis agent."
                        user_prompt = prompt

                    custom_id = f"{contract_id}__chunk__{idx}"
                    
                    request_obj = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "max_tokens": 4000
                        }
                    }
                    
                    tmp_file.write(json.dumps(request_obj) + "\n")
                    
            tmp_file_path = tmp_file.name
        finally:
            tmp_file.close()
            
        return tmp_file_path

    def submit_batch(self, jsonl_file_path: str) -> str:
        """Uploads the jsonl and creates the batch job in OpenAI."""
        client = self.get_raw_openai_client()
        
        logger.info(f"Uploading batch file: {jsonl_file_path}")
        with open(jsonl_file_path, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
            
        logger.info(f"Creating batch job with file ID: {batch_input_file.id}")
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Clause Extraction Bulk"}
        )
        return batch_job.id

    def check_status(self, batch_id: str) -> dict[str, Any]:
        """Returns the current status of the batch job."""
        client = self.get_raw_openai_client()
        batch_job = client.batches.retrieve(batch_id)
        return {
            "id": batch_job.id,
            "status": batch_job.status,
            "output_file_id": batch_job.output_file_id,
            "error_file_id": batch_job.error_file_id,
            "failed_at": batch_job.failed_at,
            "completed_at": batch_job.completed_at
        }

    def process_completed_batch(self, batch_id: str, contracts_text_map: dict[str, str]) -> dict[str, ClauseExtractorOutput]:
        """
        Downloads the completed batch output, groups responses by contract_id,
        and reconstructs the full ClauseExtractorOutput for each contract.
        
        contracts_text_map: Dict mapping contract_id to the full contract_text.
        """
        client = self.get_raw_openai_client()
        status_info = self.check_status(batch_id)
        
        output_file_id = status_info.get("output_file_id")
        if not output_file_id:
            logger.warning(f"Batch {batch_id} has no output_file_id. Status: {status_info['status']}")
            return {}

        response = client.files.content(output_file_id)
        file_content = response.text
        
        # Group by contract_id
        contract_chunks: dict[str, list[dict]] = {}
        for line in file_content.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if not custom_id:
                continue
            
            # format: {contract_id}__chunk__{idx}
            parts = custom_id.split("__chunk__")
            if len(parts) != 2:
                continue
            
            c_id = parts[0]
            if c_id not in contract_chunks:
                contract_chunks[c_id] = []
                
            response_body = data.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                content = choices[0]["message"]["content"]
                contract_chunks[c_id].append(content)

        # Reconstruct output per contract
        results = {}
        for c_id, contents in contract_chunks.items():
            clauses = []
            metadata_obj = ContractMetadata()
            full_text = contracts_text_map.get(c_id, "")
            
            for content in contents:
                parsed = _parse_llm_response(content)
                if not parsed:
                    continue
                
                new_clauses = _build_clauses_from_llm(parsed.get("clauses", []))
                clauses.extend(new_clauses)
                
                if "metadata" in parsed:
                    metadata_dict = parsed["metadata"]
                    # We need a small patch since _merge_metadata expects a ContractMetadata object as first arg
                    # Wait, _merge_metadata actually takes (ContractMetadata, dict) -> ContractMetadata. Yes.
                    metadata_obj = _merge_metadata(metadata_obj, metadata_dict)
            
            # Resolve page numbers
            if full_text:
                for clause in clauses:
                    if not clause.page_number:
                        clause.page_number = get_page_number_for_text(full_text, clause.raw_text)
                    for sub in clause.subclauses:
                        if not sub.page_number:
                            sub.page_number = get_page_number_for_text(full_text, sub.raw_text)
            
            labels = _build_cuad_labels(clauses)
            
            results[c_id] = ClauseExtractorOutput(
                clauses=clauses,
                metadata=metadata_obj,
                cuad_labels=labels,
                is_extraction_complete=True,
                extraction_completeness_notes="Extracted via Batch API."
            )
            
        return results
