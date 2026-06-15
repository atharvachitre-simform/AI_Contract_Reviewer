import json

with open('logs/checkpoints/46ce978f8a9180f4.json', 'r') as f:
    data = json.load(f)

# The keys in the checkpoint JSON
print("Keys:", list(data.keys()))
if 'contract_text' in data:
    print("Contract text length:", len(data['contract_text']))
    # Let's save the contract text to a temporary text file so we can analyze it
    with open('scratch/contract_46ce978f8a9180f4.txt', 'w') as out:
        out.write(data['contract_text'])
elif 'cleaned_text' in data:
    print("Cleaned text length:", len(data['cleaned_text']))
