import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests must depend neither on the local .env nor on any external service:
# environment variables take precedence over the .env file.
os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="", LLM_PROVIDER="ollama")
