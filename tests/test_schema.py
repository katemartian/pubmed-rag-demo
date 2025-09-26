import json
import pandas as pd
from jsonschema import validate

def test_sample_dataset_matches_schema():
    # load schema
    with open("contracts/schema.json") as f:
        schema = json.load(f)

    # load csv
    df = pd.read_csv("sample_dataset/sample.csv")

    # validate each row
    for record in df.to_dict(orient="records"):
        validate(instance=record, schema=schema)
