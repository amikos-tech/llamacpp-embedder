#!/bin/bash

while true; do
  curl  http://localhost:8080/embed_texts -d '{"model": "all-MiniLM-L6-v2.Q4_0.gguf","texts":["hello world"]}'
  sleep 0.02
done
