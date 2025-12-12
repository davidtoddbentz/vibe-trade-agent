"""Generate JSON schema from Pydantic model for LangSmith storage.

This script generates the JSON schema that should be stored in LangSmith
prompt metadata for the formatter prompt.
"""

import json

# Import directly without going through __init__.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "graph"))

from models import FormattedQuestions

if __name__ == "__main__":
    schema = FormattedQuestions.model_json_schema()
    print("JSON Schema for FormattedQuestions:")
    print(json.dumps(schema, indent=2))
    print("\n\nThis schema should be stored in LangSmith prompt metadata")
    print("under the key 'output_schema' or 'schema' for the 'formatter' prompt.")
