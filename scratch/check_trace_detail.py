with open("logs/langfuse_events.jsonl", "r") as f:
    for line in f:
        if "079264f784b132a071e253d38be3ac64" in line:
            import json

            obj = json.loads(line)
            if obj.get("step") == "generation:clause_extractor":
                print(f"Timestamp: {obj.get('timestamp')}")
                print(
                    f"Input tokens: {obj.get('input_tokens')}, Output tokens: {obj.get('output_tokens')}"
                )
                # Print payload or any details if present
                print(f"Keys: {list(obj.keys())}")
                print(f"Model: {obj.get('model')}")
                print("-" * 50)
