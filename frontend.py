import json
from datetime import datetime

import requests
import streamlit as st


API_URL = "http://localhost:8000/upload-pdf-v2"


st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")

st.title("📄 PDF Summarizer")
st.caption("Upload a PDF, send it to the backend, and inspect masked/unmasked summaries with timing metrics.")


with st.sidebar:
    st.header("Backend")
    api_url = st.text_input("Upload endpoint", value=API_URL)
    st.write("Expected method: **POST**")
    st.write("Expected payload: **multipart/form-data** with a PDF file field")
    st.divider()
    st.write("Current time:")
    st.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])


col1, col2 = st.columns([1, 1])
with col1:
    send_button = st.button("Upload and summarize", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear output", use_container_width=True)


if clear_button:
    st.session_state.pop("api_result", None)
    st.session_state.pop("raw_response", None)
    st.rerun()


if send_button:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    else:
        try:
            with st.spinner("Uploading and generating summary..."):
                files = {
                    "pdf": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                }
                response = requests.post(api_url, files=files, timeout=600)

            st.session_state["raw_response"] = response.text

            try:
                st.session_state["api_result"] = response.json()
            except Exception:
                st.session_state["api_result"] = None

            if response.ok:
                st.success(f"Request completed successfully (HTTP {response.status_code}).")
            else:
                st.error(f"Backend returned HTTP {response.status_code}.")

        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")


result = st.session_state.get("api_result")
raw_response = st.session_state.get("raw_response")


if result:
    masked_summary = result.get("masked_summary", "")
    unmasked_summary = result.get("unmasked_summary", "")
    mapping = result.get("mapping", {})
    metrics = result.get("metrics", {})

    metric_cols = st.columns(4)
    metric_cols[0].metric("Extract seconds", f"{metrics.get('extract_seconds', '-')}")
    metric_cols[1].metric("Mask seconds", f"{metrics.get('mask_seconds', '-')}")
    metric_cols[2].metric("Chunk seconds", f"{metrics.get('chunk_seconds', '-')}")
    metric_cols[3].metric("Map seconds", f"{metrics.get('map_seconds', '-')}")

    metric_cols_2 = st.columns(4)
    metric_cols_2[0].metric("Reduce seconds", f"{metrics.get('reduce_seconds', '-')}")
    metric_cols_2[1].metric("Unmask seconds", f"{metrics.get('unmask_seconds', '-')}")
    metric_cols_2[2].metric("Total seconds", f"{metrics.get('total_seconds', '-')}")
    metric_cols_2[3].metric("Chunk count", f"{metrics.get('chunk_count', '-')}")

    st.metric("Batch count", f"{metrics.get('batch_count', '-')}" )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Unmasked response",
        "Masked response",
        "Mapping",
        "Metrics",
        "Raw JSON",
    ])

    with tab1:
        if unmasked_summary:
            st.subheader("Unmasked summary")
            st.write(unmasked_summary)
        else:
            st.info("No unmasked summary found in the response.")

    with tab2:
        if masked_summary:
            st.subheader("Masked summary")
            st.write(masked_summary)
        else:
            st.info("No masked summary found in the response.")

    with tab3:
        if mapping:
            st.subheader("Mask mapping")
            st.json(mapping)
        else:
            st.info("No mapping found in the response.")

    with tab4:
        st.subheader("All metrics")
        st.json(metrics)
        st.markdown(
            f"**Chunk count:** {metrics.get('chunk_count', '-')}  \n"
            f"**Batch count:** {metrics.get('batch_count', '-')}"
        )

    with tab5:
        st.subheader("Parsed response")
        st.json(result)

elif raw_response:
    st.warning("The backend response was not valid JSON.")
    st.code(raw_response, language="text")


with st.expander("Request format expected by the backend", expanded=False):
    st.markdown(
        """
        This frontend sends:

        ```python
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
        }
        requests.post("http://localhost:8000/upload-pdf-v2", files=files)
        ```

        If your FastAPI endpoint expects a different form field name, change `"file"` above.

        The response is expected to contain:
        - `masked_summary`
        - `unmasked_summary`
        - `mapping`
        - `metrics` with:
          - `extract_seconds`
          - `mask_seconds`
          - `chunk_seconds`
          - `map_seconds`
          - `reduce_seconds`
          - `unmask_seconds`
          - `total_seconds`
          - `chunk_count`
          - `batch_count`
        """
    )
