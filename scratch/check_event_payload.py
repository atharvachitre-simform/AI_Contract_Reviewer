import json

with open("logs/langfuse_events.jsonl", "r") as f:
    for line in f:
        if "079264f784b132a071e253d38be3ac64" in line:
            obj = json.loads(line)
            # Find generation events
            if "generation" in obj.get("step", ""):
                print(f"Step: {obj.get('step')}")
                print(f"Timestamp: {obj.get('timestamp')}")
                print(f"Input: {obj.get('input_tokens')}, Output: {obj.get('output_tokens')}")
                # We can't see the output payload directly since it's not in the event log keys,
                # but let's print all payload fields if they exist
                if "payload" in obj:
                    print(f"Payload: {obj['payload'][:200]}")
                print("-" * 60)
