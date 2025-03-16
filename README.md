SDR:
an api to apply machine intelligence to historical data

api:
/ingest
takes an image (e.g. of a newspaper) and puts it in a queue for digesting

/search
searches the index

Also an MCP server for a deepresearch style frontend

data pipeline:

1. raw data

- jsonl, each row contains metadata and path to file

2. segmentation

- updates each row with paths to segments and positions on page

3. ocr

- updates each row with transcription of each segment

4. llm rewriting -> articles

- given a succession of pages, extract articles and make a new index

5. llm summarizing articles

all llm stuff is done with vllm

the atomic unit of data is the article
