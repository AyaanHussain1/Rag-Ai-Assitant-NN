import streamlit as st

from Processing_user_query import answer_query, build_uploaded_course_index

st.set_page_config(page_title="Course Companion", page_icon="✦", layout="wide")
st.markdown("""<style>
.stApp{background:#111827;color:#f9fafb}.block-container{max-width:940px;padding-top:2rem}
[data-testid="stHeader"]{background:transparent}.brand{font-size:1.05rem;font-weight:650;letter-spacing:-.02em;color:#f9fafb}
.hero{text-align:center;margin-top:18vh}.hero h1{font-size:clamp(2rem,4vw,3.3rem);letter-spacing:-.055em;margin-bottom:.4rem;color:#f9fafb}.hero p{color:#9ca3af;font-size:1.05rem;margin-bottom:1.6rem}
.stTextArea textarea{background:#1f2937!important;color:#f9fafb!important;border:1px solid #374151!important;border-radius:18px!important;box-shadow:0 2px 12px rgba(0,0,0,.22)!important;padding:16px!important;font-size:1rem!important}.stTextArea textarea::placeholder{color:#9ca3af!important}
.stTextArea textarea:focus{border-color:#6b7280!important;box-shadow:0 3px 18px rgba(0,0,0,.34)!important}.stButton button{border-radius:999px;background:#f9fafb;color:#111827;border:0;padding:.55rem 1.1rem}.stButton button:hover{background:#d1d5db;color:#111827}.hint{color:#6b7280;font-size:.82rem;text-align:center;margin-top:.4rem}.message-label{color:#9ca3af;font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-top:1.5rem}.stFileUploader{color:#d1d5db}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="brand">✦ Course Companion</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "uploaded_course_index" not in st.session_state:
    st.session_state.uploaded_course_index = None

if st.session_state.messages:
    st.markdown("### Your course conversation")
    for message in st.session_state.messages:
        label = "User" if message["role"] == "user" else "Instructor"
        st.markdown(f'<div class="message-label">{label}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(message["content"])
else:
    st.markdown('<div class="hero"><h1>How can I help with your video?</h1><p>Give me any course video and I’ll help you find and understand what you need.</p></div>', unsafe_allow_html=True)

left, upload_column, right = st.columns([2.2, 1, 2.2])
with upload_column:
    if st.button("＋ Add video", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader

if st.session_state.show_uploader:
    uploaded_files = st.file_uploader(
        "Upload transcript JSON files",
        type=["json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption("Upload the JSON transcript files created from your videos. They are embedded for this chat session.")
    if uploaded_files:
        st.success(f"{len(uploaded_files)} JSON file(s) selected")
        if st.button("Prepare uploaded videos"):
            with st.spinner("Creating a searchable course index..."):
                try:
                    st.session_state.uploaded_course_index = build_uploaded_course_index(uploaded_files)
                    st.success("Uploaded videos are ready. Your next questions will use their content.")
                except (RuntimeError, ValueError) as error:
                    st.error(f"Could not prepare the uploaded files: {error}")

with st.form("query_form", clear_on_submit=True):
    query = st.text_area("Ask a question", placeholder="e.g. Where is backpropagation explained?", label_visibility="collapsed", height=92)
    _, submit_column, _ = st.columns([2.2, 1, 2.2])
    with submit_column:
        submitted = st.form_submit_button("Ask course")

st.markdown('<div class="hint">Upload a video or ask a course question · Powered by your local models</div>', unsafe_allow_html=True)

if submitted:
    if not query.strip():
        st.warning("Please type a course question first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query.strip()})
        with st.spinner("Finding the best lesson for you..."):
            try:
                response = answer_query(
                    query.strip(), retrieval_df=st.session_state.uploaded_course_index
                )
            except (RuntimeError, ValueError) as error:
                response = f"I couldn't complete that request: {error}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
