# Content Summarization API (MVP)

This project provides a simple content summarization service using a pre-trained transformer model (BART). You can paste text into a webpage, click **Summarize**, and get bullet-pointed summaries.

## Features

- **Summarization of Long Texts:**  
  Splits large inputs into manageable chunks and returns multiple summaries if needed.
  
- **Pre-trained BART Model:**  
  Uses `facebook/bart-large-cnn` from Hugging Face.
  
- **Simple Frontend UI:**  
  An `index.html` page for direct interaction without complex tooling.

## Requirements

- **Python 3.8+**
- **Dependencies in `requirements.txt`:**
  - fastapi
  - uvicorn[standard]
  - transformers
  - pydantic

Install them:
```bash
pip install -r requirements.txt
```

## Getting Started
1. Clone the Repository:

```bash
git clone https://github.com/yourusername/content-summarization-api.git
cd content-summarization-api
```
2. Set Up a Virtual Environment (Recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn app:app --reload
```
The API runs at http://127.0.0.1:8000.

4. Serve the Frontend: In another terminal:

```bash
python3 -m http.server 8080
```
Open http://127.0.0.1:8080/index.html in your browser, paste text, and click Summarize.

## Usage
API Endpoint: POST /summarize

Request JSON:

```json
{
  "text": "Your text here",
  "max_length": 150,
  "min_length": 30
}
```
Response JSON:

```json
{
  "summaries": [
    "Summary chunk 1",
    "Summary chunk 2"
  ]
}
```
## Troubleshooting

- CORS or Origin Issues:
  - If needed, enable CORS in app.py.
  - Always access index.html via http://127.0.0.1:8080, not file://.

- Model Downloading:
  - On first run, Transformers will download the BART model. Ensure a stable internet connection.

- Input Size:
  - Very long text is chunked automatically. Just wait a bit longer for results.

## Contributing
Feel free to open issues or submit PRs for improvements.

## License
Licensed under the MIT License. See the LICENSE file for details.