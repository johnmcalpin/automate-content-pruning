import os
import re
import csv
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd

@st.cache
def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    body = soup.body
    for tag in body(['header', 'nav', 'footer']):
        tag.decompose()
    return body.text

@st.cache
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    return text

@st.cache
def get_similarity_matrix(documents):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

@st.cache
def cluster_documents(similarity_matrix, threshold):
    cluster_number = 0
    clusters = []
    visited = [False] * len(similarity_matrix)

    for i in range(len(similarity_matrix)):
        if visited[i]:
            continue

        cluster = []
        dfs(i, visited, cluster, similarity_matrix, threshold)
        clusters.append(cluster)
        cluster_number += 1

    return clusters

@st.cache
def dfs(i, visited, cluster, similarity_matrix, threshold):
    visited[i] = True
    cluster.append(i)

    for j in range(len(similarity_matrix)):
        if visited[j]:
            continue

        if similarity_matrix[i][j] >= threshold:
            dfs(j, visited, cluster, similarity_matrix, threshold)

def app():
    st.title("Document Cluster Analysis")

    with open("urls.txt", "r") as file:
        urls = file.read().splitlines()

    documents = [preprocess_text(get_text_from_url(url)) for url in urls]

    similarity_matrix = get_similarity_matrix(documents)

    clusters = cluster_documents(similarity_matrix, 0.5)

    cluster_dict = {"Cluster": [], "URL": []}
    for cluster_index, cluster in enumerate(clusters):
        for document_index in cluster:
            cluster_dict["Cluster"].append(cluster_index)
            cluster_dict["URL"].append(urls[document_index])

    df = pd.DataFrame(cluster_dict)
    
    st.table(df)

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="clusters.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
