import streamlit as st
import json

st.session_state.setdefault("submitted", False)

with open('questions.json') as f:
    data = json.load(f)

st.title("Questionnaire")

if not st.session_state["submitted"]:
    disorder = 'Anxiety'
    
    question_placeholder = st.empty()
    
    with question_placeholder.container():
        for i, question in enumerate(data[disorder]):
            st.write(question["question"])
            st.radio("", question["options"], key=f"question_{i}")
        
        if st.button("Submit"):
            responses = [st.session_state.get(f"question_{i}") for i in range(len(data[disorder]))]
            st.session_state["submitted"] = True
            question_placeholder.empty()

if st.session_state["submitted"]:
    st.write("Thank you! Your response has been saved.")
