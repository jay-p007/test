import json
import os
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="Document Intelligence UI", page_icon="DOC", layout="wide")

API_DEFAULT = os.getenv("BACKEND_URL", "https://document-intelligence-system-2xuf.onrender.com")

if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {"title": "New Chat", "session_id": first_id, "messages": []}
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = next(iter(st.session_state.chats.keys()))
if "rename_buffer" not in st.session_state:
    st.session_state.rename_buffer = ""


def render_citations(citations):
    st.markdown("### Citations")
    if not citations:
        st.info("No citations returned.")
        return
    for i, c in enumerate(citations, start=1):
        with st.expander(
            f"{i}. {c['filename']} | page {c['page']} | chunk {c['chunk_id']} | score {c['score']}"
        ):
            st.write(c["excerpt"])


def render_agent_trace(trace):
    st.markdown("### Agent Trace")
    if not trace:
        st.info("No agent trace returned.")
        return
    for step in trace:
        st.write(f"- {step}")

st.title("Document Intelligence AI System")
st.caption("Upload PDFs and ask grounded questions using the FastAPI backend.")

with st.sidebar:
    st.header("Chats")

    if st.button("+ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "session_id": new_id, "messages": []}
        st.session_state.active_chat_id = new_id
        st.session_state.rename_buffer = ""

    for chat_id, chat_data in list(st.session_state.chats.items()):
        is_active = chat_id == st.session_state.active_chat_id
        label = f"* {chat_data['title']}" if is_active else chat_data["title"]
        if st.button(label, key=f"chat_btn_{chat_id}", use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.session_state.rename_buffer = chat_data["title"]

    active_chat = st.session_state.chats[st.session_state.active_chat_id]
    st.caption("Manage Active Chat")
    st.session_state.rename_buffer = st.text_input(
        "Rename",
        value=st.session_state.rename_buffer or active_chat["title"],
        key="rename_input",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Name", use_container_width=True):
            name = st.session_state.rename_buffer.strip()
            active_chat["title"] = name or "New Chat"
    with c2:
        if st.button("Delete Chat", use_container_width=True, disabled=len(st.session_state.chats) <= 1):
            del st.session_state.chats[st.session_state.active_chat_id]
            st.session_state.active_chat_id = next(iter(st.session_state.chats.keys()))
            st.session_state.rename_buffer = st.session_state.chats[st.session_state.active_chat_id]["title"]

    st.divider()
    st.header("Settings")
    api_base = st.text_input("Backend URL", value=API_DEFAULT).rstrip("/")
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=5)
    use_stream = st.toggle("Use Streaming", value=True)
    show_raw_json = st.toggle("Show Raw JSON", value=False)
    model_profile_label = st.selectbox(
        "Model Profile",
        options=["low_cost", "balanced", "high_quality", "custom"],
        index=1,
        help="Use low_cost for cheaper calls, balanced for normal use, high_quality for better output.",
    )
    custom_model = st.text_input("Custom Gemini Model", value="gemini-2.5-flash") if model_profile_label == "custom" else ""

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Upload Document", use_container_width=True, disabled=uploaded_file is None):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{api_base}/upload", files=files, timeout=240)
            response.raise_for_status()
            payload = response.json()
            st.session_state.documents[payload["document_id"]] = payload
            ocr_text = "Yes" if payload.get("used_ocr") else "No"
            st.success(
                f"Uploaded: {payload['filename']} | pages={payload['pages']} | chunks={payload['chunks']} | OCR={ocr_text}"
            )
            if show_raw_json:
                st.markdown("### Raw Upload JSON")
                st.json(payload)
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Upload failed. {detail}")

with col2:
    st.subheader("2) Ask Question")
    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    if active_chat["messages"]:
        st.markdown("### Chat History")
        for msg in active_chat["messages"]:
            role = "You" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {msg['content']}")

    options = ["All indexed documents"]
    option_to_doc_id = {"All indexed documents": None}
    for doc_id, info in st.session_state.documents.items():
        label = f"{info['filename']} ({doc_id[:8]})"
        options.append(label)
        option_to_doc_id[label] = doc_id

    selected = st.selectbox("Document scope", options=options)
    question = st.text_area("Question", placeholder="Ask question based on the uploaded document")

    if st.button("Ask", use_container_width=True, disabled=not question.strip()):
        session_id = active_chat.get("session_id") or st.session_state.active_chat_id
        model_profile = "balanced" if model_profile_label == "custom" else model_profile_label
        body = {
            "question": question.strip(),
            "document_id": option_to_doc_id[selected],
            "top_k": top_k,
            "session_id": session_id,
            "model_profile": model_profile,
            "generation_model": custom_model or None,
        }
        active_chat["messages"].append({"role": "user", "content": question.strip()})
        if active_chat["title"] == "New Chat":
            active_chat["title"] = question.strip()[:40]

        if not use_stream:
            try:
                response = requests.post(f"{api_base}/ask", json=body, timeout=240)
                response.raise_for_status()
                data = response.json()
                active_chat["session_id"] = data.get("session_id", session_id)
                answer = data.get("answer", "")
                active_chat["messages"].append({"role": "assistant", "content": answer})
                st.markdown("### Answer")
                st.write(answer)
                structured = data.get("structured_answer", {})
                if structured:
                    st.markdown("### Structured Answer")
                    st.json(structured)
                st.info(f"Model Used: {data.get('model_used', 'n/a')}")
                st.metric("Confidence", f"{data.get('confidence', 0):.4f}")
                render_agent_trace(data.get("agent_trace", []))
                render_citations(data.get("citations", []))
                if show_raw_json:
                    st.markdown("### Raw Ask JSON")
                    st.json(data)
            except requests.RequestException as exc:
                detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
                st.error(f"Question failed. {detail}")
        else:
            try:
                with requests.post(f"{api_base}/ask/stream", json=body, timeout=240, stream=True) as response:
                    response.raise_for_status()
                    answer_placeholder = st.empty()
                    model_placeholder = st.empty()
                    answer_accum = ""
                    meta = None
                    model_used = ""
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line or not raw_line.startswith("data: "):
                            continue
                        event = json.loads(raw_line[6:])
                        if event.get("type") == "meta":
                            meta = event
                            active_chat["session_id"] = event.get("session_id", session_id)
                        elif event.get("type") == "model":
                            model_used = event.get("model_used", "")
                            model_placeholder.info(f"Model Used: {model_used}")
                        elif event.get("type") == "chunk":
                            answer_accum += event.get("text", "")
                            answer_placeholder.markdown("### Answer\n" + answer_accum)
                        elif event.get("type") == "done":
                            break
                    active_chat["messages"].append({"role": "assistant", "content": answer_accum.strip()})
                    if meta and not model_used:
                        model_placeholder.info("Model Used: n/a")
                    if meta:
                        st.metric("Confidence", f"{meta.get('confidence', 0):.4f}")
                        render_agent_trace(meta.get("agent_trace", []))
                        render_citations(meta.get("citations", []))
                        if show_raw_json:
                            st.markdown("### Raw Stream Meta JSON")
                            st.json(meta)
            except requests.RequestException as exc:
                detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
                st.error(f"Streaming failed. {detail}")

st.divider()
st.markdown("Run backend first: `uvicorn app.main:app --reload` and UI: `streamlit run streamlit_app.py`")
