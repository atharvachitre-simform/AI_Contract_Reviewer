import asyncio
from src.services.services import ContractReviewService
from pathlib import Path

def main():
    service = ContractReviewService()
    contract_id = "46ce978f8a9180f4"
    text = Path("scratch/contract_46ce978f8a9180f4.txt").read_text()
    
    print(f"Processing contract {contract_id} with visualizer ON...")
    result = service.process_contract(
        contract_text=text,
        contract_id=contract_id
    )
    print("Done!")

if __name__ == "__main__":
    main()
