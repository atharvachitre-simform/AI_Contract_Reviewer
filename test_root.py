from dotenv import load_dotenv
load_dotenv(".env")
from langfuse import Langfuse
client = Langfuse()
obs = client.start_observation(as_type="span", name="root_span_test", user_id="test_user")
print("Trace ID:", obs.trace_id)
obs.end()
client.flush()
