import os

session_id = "46ce978f8a9180f4"
for root, dirs, files in os.walk("logs/chat"):
    for file in files:
        if session_id in file or "summary" in file or "chat_history" in file:
            path = os.path.join(root, file)
            print(f"File: {path}")
            if file.endswith(".txt") or file.endswith(".json"):
                with open(path, "r") as f:
                    print(f.read()[:1000])
                    print("=" * 80)
