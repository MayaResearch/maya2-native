#!/usr/bin/env python3
"""Use the OpenAI Python SDK against a running maya3.server (pip install openai).

    python examples/openai_client.py --base-url http://127.0.0.1:8080/v1
"""
import argparse

from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
a = ap.parse_args()
client = OpenAI(base_url=a.base_url, api_key="unused")
with client.audio.speech.with_streaming_response.create(
        model="maya3", voice="Kayal", input="வணக்கம்! உங்கள் பதிவு உறுதிசெய்யப்பட்டது.",
        response_format="wav", extra_body={"language": "ta"}) as r:
    r.stream_to_file("openai_ta.wav")
print("wrote openai_ta.wav")
