import streamlit as st
import json
import os
# from sentence_transformers import SentenceTransformer
# import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tashaphyne.stemming import ArabicLightStemmer
from googletrans import Translator

# Initialize the Arabic light stemmer
stemmer = ArabicLightStemmer()

# Initialize the translator
translator = Translator()

def normalize_text(text):
    # Remove "ال" from the beginning of words
    text = re.sub(r'\bال', '', text)
    # Replace "تاء" with "هاء"
    text = text.replace('ة', 'ه')

    # Normalize all forms of alef to simple "ا"
    text = re.sub(r'[أإآا]', 'ا', text)

    # Remove all Tashkeel (Arabic diacritics)
    tashkeel = r'[ًٌٍَُِّْ]'
    text = re.sub(tashkeel, '', text)

    return text

# Exact Match Search Function
def exact_match_search(query, sentences):
    normalized_query = normalize_text(query)
    
    # Normalize sentences
    normalized_sentences = [normalize_text(sentence) for sentence in sentences]
    
    # Check if the normalized query is a substring of any sentence
    matched_sentences = [
        json_data[i] for i, sentence in enumerate(normalized_sentences)
        if normalized_query in sentence
    ]
    
    # If matches are found, return them with exact=True
    if matched_sentences:
        return matched_sentences, True
    else:
        return None, False


# Function to translate text from Arabic to English
def translate_arabic_to_english(text):
    # Perform the translation
    translation = translator.translate(text, src='ar', dest='en')
    return translation.text

# Function to preprocess the Arabic text (with stemming + removing punctuation)
def preprocess_text(text):
    text = normalize_text(text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split the text into words
    words = text.split()
    
    # Apply stemming to each word
    stemmed_words = [stemmer.light_stem(word) for word in words]
    
    # Rejoin the stemmed words into a single sentence
    return ' '.join(stemmed_words)

# Load data from JSON
with open("data.json", "r") as f:
    data = json.load(f)

# print("Load the semantic model.")
# model = SentenceTransformer("Omartificial-Intelligence-Space/mpnet-base-all-nli-triplet-Arabic-mpnet_base")

# Load the JSON data from a file
json_file_path = "common_voice_metadata.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

# Extract the sentences from the JSON data
sentences = [item['sentence'] for item in json_data]


# Preprocess all sentences (with stemming)
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Create a TF-IDF Vectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the preprocessed sentences and transform them to vectors
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

# Function to get TF-IDF similarity
def get_tfidf_similarity(query, tfidf_matrix):
    # Preprocess the query using the same stemming technique
    preprocessed_query = preprocess_text(query)
    
    # Vectorize the query using the TF-IDF vectorizer
    query_tfidf = vectorizer.transform([preprocessed_query])
    
    # Compute cosine similarities between the query and the sentence vectors
    tfidf_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    return tfidf_similarities

# Function to find the most similar sentence using TF-IDF similarity
def find_most_similar_sentence(query, preprocessed_sentences, tfidf_matrix, threshold=0.2):
    # Get TF-IDF similarities
    tfidf_similarities = get_tfidf_similarity(query, tfidf_matrix)
    
    # Apply threshold to filter out low-scoring sentences
    filtered_indices = [i for i, score in enumerate(tfidf_similarities) if score >= threshold]
    
    # Sort the sentences by their similarity scores (descending order)
    sorted_indices = sorted(filtered_indices, key=lambda i: tfidf_similarities[i], reverse=True)
    
    # Return the top result(s)
    if sorted_indices:
        top_index = sorted_indices[0]
        return json_data[top_index], tfidf_similarities[top_index]
    else:
        return None, 0  # No result meets the threshold



# print("Load Embeddings")
# embeddings = torch.load("embds.pt")

# Helper function to search text
def search_by_text(query):

    query = normalize_text(query)

    exact_match_results, exact = exact_match_search(query, sentences)
        
    if exact_match_results:
        # If exact match is found, return it with `exact=True`
        return exact_match_results, exact

    tokens = query.split()
    if len(tokens)>1:
        exact_match_results = []
        for token in tokens:
        # First, try exact match search
            exact_match_result, exact = exact_match_search(token, sentences)
            
            if exact_match_result:
                exact_match_results.extend(exact_match_result)
                # If exact match is found, return it with `exact=True`
        if len(exact_match_results)>0:
            return exact_match_results, True
            
    
    # Find the most similar sentence
    most_similar_sentence, score = find_most_similar_sentence(query, preprocessed_sentences, tfidf_matrix)

    # Output the result
    if most_similar_sentence:
        return [most_similar_sentence], False
        # print(f"Most similar sentence: {most_similar_sentence}")
        # print(f"Similarity score: {score}")
    else:
        print("No sentence meets the threshold.")
        return [], False
    # embeddings_query = model.encode(query)
    # similarities = torch.matmul(torch.tensor(embeddings), torch.tensor(embeddings_query).T)
    # largest_similarity_index = torch.argmax(similarities)

    # print(similarities[largest_similarity_index])
    # return [json_data[largest_similarity_index]]

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
st.sidebar.title("AliveArchive")
search_option = st.sidebar.radio("Search by:", ("Text", "Speech"))

MAX_OUTPUT=10
# Main page content based on search option
if search_option == "Text":
    st.title("Text Search")

    # Form to capture text and submit on Enter
    with st.form(key="search_form"):
        query = st.text_input("Enter your search query:")
        submit_button = st.form_submit_button(label="Search")

    if submit_button:
        results, exact = search_by_text(query)
        results = results[:MAX_OUTPUT]
        if results:
            if exact:
                st.write(f"Found {len(results)} exact match result(s):")
            else:
                st.write(f"Found {len(results)} similar result(s):")
            for i, result in enumerate(results):
                st.write(f"**Result {i + 1}:**")
                path = result['path'].replace("/home/vscode/.cache/huggingface/datasets/downloads/extracted/806b3f94e57426271901dfd6c41899e3ae63486dfd11f50e0a2d6d3c9bc0e090/", "./")
                st.audio(path)
                english_translation = translate_arabic_to_english(result['sentence'])
                st.json(
                    {
                        "Transcription": result['sentence'],
                        "Translation": english_translation,
                        "Gender": result['gender'],
                        "Age": result['age'],
                    },
                    expanded=2,
                )
                # st.write(f"Transcription: {result['sentence']}")
                
                # st.write(f"Translation: {english_translation}")
                # st.write(f"Gender: {result['gender']}")
                # st.write(f"Age: {result['age']}")
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
