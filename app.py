import base64
import json
import urllib.request
import urllib.error
import urllib.parse
import streamlit as st
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Image Describer (Ollama)", layout="centered")
st.title("Paveikslėlio apibūdinimas (Ollama)")

st.markdown("Ši programa leidžia įkelti paveikslėlį ir apibūdina jo turinį naudojant lokalią Ollama instanciją. "
			"Jei `ollama` Python paketas nėra įdiegtas, bus panaudotas HTTP fallback (pvz., http://localhost:11434).")

# (No alias mappings — app will use the exact model name entered in the sidebar)

# Model discovery removed: the app will use the exact model name entered in the sidebar.

with st.sidebar:
	st.header("Konfigūracija")
	ollama_host = st.text_input("Ollama hostas (HTTP)", value="http://localhost:11434")
	ollama_model = st.text_input("Modelis", value="gemma3:4b")
	st.markdown("\n")

uploaded_file = st.file_uploader("Įkelkite paveikslėlį", type=["png", "jpg", "jpeg"])


def call_ollama_http(image_bytes: bytes, prompt: str, host: str, model: str) -> str:
	host = host.rstrip("/")
	endpoints = ["/chat", "/api/chat", "/v1/chat", "/generate", "/api/generate", "/v1/generate"]

	image_b64 = base64.b64encode(image_bytes).decode("utf-8")

	payload_body = {
		"model": model,
		"messages": [{
			"role": "user",
			"content": prompt,
			"images": [image_b64]
		}]
	}

	payload_no_model = {
		"messages": [{
			"role": "user",
			"content": prompt,
			"images": [image_b64]
		}]
	}

	last_exc = None
	for ep in endpoints:
		for use_model_in_query in (True, False):
			if use_model_in_query:
				url = f"{host}{ep}?model={urllib.parse.quote(model, safe='') }"
				data = json.dumps(payload_no_model).encode("utf-8")
			else:
				url = f"{host}{ep}"
				data = json.dumps(payload_body).encode("utf-8")

			req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
			try:
				with urllib.request.urlopen(req, timeout=30) as resp:
					body = resp.read()
					try:
						parsed = json.loads(body)
					except Exception:
						return body.decode("utf-8", errors="replace")

					if isinstance(parsed, dict):
						if "message" in parsed and isinstance(parsed["message"], dict) and parsed["message"].get("content"):
							return parsed["message"]["content"]
						if parsed.get("response"):
							return parsed.get("response")
					return json.dumps(parsed, ensure_ascii=False)

			except urllib.error.HTTPError as he:
				try:
					err_body = he.read().decode("utf-8", errors="replace")
				except Exception:
					err_body = ""
				last_exc = RuntimeError(f"HTTP error from Ollama: {he.code} {he.reason} - {err_body}")
			except urllib.error.URLError as ue:
				last_exc = RuntimeError(f"Could not connect to Ollama HTTP API: {ue.reason}")

	if last_exc:
		raise last_exc
	raise RuntimeError(f"No reachable Ollama endpoint at {host} (tried several paths)")


if uploaded_file is not None:
	try:
		image = Image.open(uploaded_file)
		# use width='stretch' to avoid deprecation warnings in newer Streamlit
		try:
			st.image(image, width='stretch')
		except Exception:
			# fallback for older Streamlit versions
			st.image(image, use_container_width=True)

		uploaded_file.seek(0)
		image_bytes = uploaded_file.read()

		prompt = "Apibūdink šį paveikslėlį lietuvių kalba. Būk trumpas ir aiškus."

		with st.spinner("Analizuojama modeliu..."):
			# No discovery: use the exact model name provided in the sidebar
			effective_model = ollama_model

			# Prefer HTTP-first using the exact model name provided in the sidebar
			http_exc = None
			try:
				description = call_ollama_http(image_bytes, prompt, ollama_host, effective_model)
				st.subheader("Modelio atsakymas (HTTP)")
				st.write(description)
			except Exception as e_http:
				http_exc = e_http
				# Try Python client as a last resort
				try:
					from ollama import chat as _ollama_chat  # type: ignore

					response = _ollama_chat(
						model=effective_model,
						messages=[{"role": "user", "content": prompt, "images": [image_bytes]}]
					)

					description = None
					if hasattr(response, "message") and getattr(response.message, "content", None):
						description = response.message.content
					elif getattr(response, "response", None):
						description = response.response
					else:
						description = str(response)

					st.subheader("Modelio atsakymas (ollama Python klientas)")
					st.write(description)

				except Exception as e_client:
					combined = f"HTTP klaida: {http_exc}\n\nPython klientas klaida: {e_client}\n\n"
					st.error("Klaida kviečiant Ollama modelį: \n\n" + combined)
					st.info(f"Patikrinkite, ar Ollama serveris pasiekiamas: {ollama_host} ir ar modelis {effective_model} yra įdiegtas.")

	except Exception as e:
		st.error("Nepavyko atidaryti paveikslėlio: " + str(e))

st.markdown("---")
st.markdown("Reikalavimai: `streamlit`, `Pillow`. `ollama` Python paketas nebūtinas (naudosime HTTP fallback).\n\nPaleidimas: `streamlit run app.py`\n")
