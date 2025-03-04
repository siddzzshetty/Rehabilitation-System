import streamlit as st


with st.form("my_form"):
    st.write("Welcome! Your Rehabilitation Assistant is here to guide you! ")
    email = st.text_input(label = "Email", value="assisstant@gmail.com")
    password = st.text_input(label = "Password", value="pass@123")
    if st.form_submit_button():
        st.switch_page("pages/final1.py")