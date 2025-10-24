# Visa Approval Probability Assistant (Gemini + RAG)

A lightweight assistant that predicts visa approval probability using Gemini LLM and retrieval-augmented generation (RAG) over custom visa rules.

## Features

- Chat interface (Gradio) for visa questions
- Gemini 1.5 Flash API for reasoning and response
- RAG: Retrieves relevant visa rules using MiniLM embeddings + FAISS
- Upload document text for more accurate advice
- Shows top retrieved rules for transparency
- Runs on Hugging Face Spaces (free CPU tier)

## Setup

1. **Clone the repo and install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Set your Gemini API key:**
   - On Hugging Face Spaces: Add a secret named `GOOGLE_API_KEY`.
   - Locally: Set the environment variable before running:
     ```sh
     export GOOGLE_API_KEY=your-gemini-api-key
     ```

3. **Edit visa rules:**
   - Update `data/visa_rules.json` to add or modify visa rules.
   - (Re)start the app to rebuild embeddings if rules change.

## Running

- **Locally:**  
  ```sh
  python app.py
  ```
- **On Hugging Face Spaces:**  
  Push the repo and deploy. The app will auto-load and cache embeddings.

## Usage

- Enter your visa question in the chat.
- Optionally upload a `.txt` document for context.
- The assistant will estimate approval probability, list missing documents/risk factors, and give advice.
- Top retrieved rules are shown in the sidebar.

## Notes

- Embeddings and FAISS index are cached for fast startup.
- If no rules are found, Gemini will answer using its own knowledge.
- Progress bar appears when building embeddings.

## Troubleshooting

- To force a rebuild of embeddings, delete the `.pkl` and `.faiss` files in `data/`.
- Check logs for errors if the app fails to start.

---

**Made for Hugging Face Spaces.**