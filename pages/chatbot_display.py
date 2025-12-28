import streamlit as st
from st_helper import load_llm_feedback, display_sidebar

st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")
display_sidebar()
llm_messages = load_llm_feedback()

col1, col2 = st.columns([5, 2])  # Adjust column ratio as needed

# with col2:
st.subheader("Feedback")
for i, message in enumerate(llm_messages):
    if "Incorrect:" in message and "Suggestion:" in message:
        incorrect, suggestion = message.split("Suggestion:")
        if i==0:
            with st.expander(f"Feedback {i+1}", expanded=True):
                st.markdown(f":red[**Incorrect:**] {incorrect.replace('Incorrect:', '').strip()}")
                st.markdown(f":green[**Suggestion:**] {suggestion.strip()}")
        else:
            with st.expander(f"Feedback {i+1}", expanded=True):
                st.markdown(f":red[**Incorrect:**] {incorrect.replace('Incorrect:', '').strip()}")
                st.markdown(f":green[**Suggestion:**] {suggestion.strip()}")
    else:
        if i == 0:
            with st.expander(f"Feedback {i+1}", expanded=True):
                st.markdown(message)
        else:
            with st.expander(f"Feedback {i+1}"):
                st.markdown(message)
    # st.markdown(llm_messages)



