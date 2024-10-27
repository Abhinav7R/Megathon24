import streamlit as st
import json
import pickle as pkl

# Initialize responses_dict
responses_dict = {}

with open('questions.json') as f:
    data = json.load(f)

st.title("Questionnaire")

# Input for disorder list
input_disorder = st.text_input("Enter disorders (comma-separated):")

# Process the input string into a list
disorder_list = [disorder.strip() for disorder in input_disorder.split(',')]
print("disorder list:  ", disorder_list)

if not st.session_state.get("submitted", False):
    question_placeholder = st.empty()
    
    with question_placeholder.container():
        for disorder in disorder_list:
            if disorder in data:
                for i, question in enumerate(data[disorder]):
                    st.write(question["question"])
                    # Create a unique key using disorder name and question index
                    st.radio("", question["options"], key=f"{disorder}_question_{i}", label_visibility="collapsed")

        # Add a single submit button for all disorders
        if st.button("Submit All"):
            for disorder in disorder_list:
                if disorder in data:
                    responses = [question["options"].index(st.session_state.get(f"{disorder}_question_{i}")) for i, question in enumerate(data[disorder])]
                    responses_dict[disorder] = responses

            st.session_state["submitted"] = True
            question_placeholder.empty()

            # Save the responses dictionary to a file
            with open('responses.pkl', 'wb') as f:
                pkl.dump(responses_dict, f)

if st.session_state.get("submitted", False):
    st.write("Thank you! Your response has been saved.")
