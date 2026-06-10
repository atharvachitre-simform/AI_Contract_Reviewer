import sys, logging
from dotenv import load_dotenv
load_dotenv(".env")
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
from langfuse import Langfuse
client = Langfuse()
trace_id = client.create_trace_id()
ctx = {"trace_id": trace_id}
obs = client.start_observation(trace_context=ctx, as_type="generation", name="test_obs", model="test_model", input="in", output="out")
obs.end()
client.flush()
print("Trace URL:", client.get_trace_url(trace_id=trace_id))
