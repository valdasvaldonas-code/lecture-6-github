# lecture-6-github

Image Describer (Streamlit + Ollama HTTP fallback)

This small app lets you upload an image and asks a local Ollama model to describe it.

Quick start
- Create and activate a virtual environment (recommended Python 3.11 or 3.10 on Windows to avoid compiled dependency issues)
- Install dependencies: pip install -r requirements.txt
- Start the app: streamlit run app.py

Notes
- The app prefers the `ollama` Python client if it's installed. If not, it will try an HTTP fallback to the host you configure in the sidebar (default: http://localhost:11434).
- If you run into connection errors, check that the Ollama desktop/server is running and that the REST API is reachable from this machine.
