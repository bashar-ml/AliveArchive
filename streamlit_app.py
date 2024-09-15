import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
import torch

# Load data from JSON
with open("data.json", "r") as f:
    data = json.load(f)

print("Load the semantic model.")
model = SentenceTransformer("Omartificial-Intelligence-Space/mpnet-base-all-nli-triplet-Arabic-mpnet_base")

# Load the JSON data from a file
json_file_path = "common_voice_metadata.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

# Extract the sentences from the JSON data
sentences = [item['sentence'] for item in json_data]

print("Load Embeddings")
embeddings = torch.load("embds.pt")

# Helper function to search text
def search_by_text(query):
    embeddings_query = model.encode(query)
    similarities = torch.matmul(torch.tensor(embeddings), torch.tensor(embeddings_query).T)
    largest_similarity_index = torch.argmax(similarities)

    print(similarities[largest_similarity_index])
    return [json_data[largest_similarity_index]]

    # results = []
    # for item in data:
    #     if query.lower() in item['transcription'].lower():
    #         results.append(item)
    # return results

# Helper function to search by audio similarity (mock function)
def search_by_audio(audio_file):
    # Mock similarity search
    results = data[:3]  # Return first 3 results as a mock response
    return results

# Sidebar for navigation
st.sidebar.title("AliveArchive Search")
search_option = st.sidebar.radio("Search by:", ("Text", "Speech"))

# Main page content based on search option
if search_option == "Text":
    st.title("Text Search")
    query = st.text_input("Enter your search query:")
    if st.button("Search"):
        results = search_by_text(query)
        if results:
            st.write(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                # import pdb; pdb.set_trace()
                st.write(f"**Result {i + 1}:**")
                st.audio(result['path'])
                st.write(f"Transcription: {result['sentence']}")
                st.write(f"Gender: {result['gender']}")
                st.write(f"Age: {result['age']}")
                st.write("---")
        else:
            st.write("No results found.")

elif search_option == "Speech":
    st.title("Speech Search")
    uploaded_file = st.file_uploader("Upload or record an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Search by Similarity"):
            results = search_by_audio(uploaded_file)
            st.write(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                st.write(f"**Result {i + 1}:**")
                st.audio(result['audio_file'])
                st.write(f"Transcription: {result['transcription']}")
                st.write("---")

# Mock function for audio recording (optional)
if 'speech_recorder' in st.session_state:
    st.session_state['speech_recorder']()
else:
    st.write("Audio recording functionality can be added using JavaScript in Streamlit.")
