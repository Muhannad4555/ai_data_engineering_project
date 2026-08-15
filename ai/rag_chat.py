import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"

NEW_REVIEWS = 80
TOP_K = 5


CACHE_FILE = "review_embeddings_gemini.parquet"


client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"))


def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        WHERE COMMENT IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {NEW_REVIEWS}
    """

    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df

def embed_texts(texts, task_type):
    
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )

    return [embedding.values for embedding in response.embeddings]


def embed_in_batches(texts, task_type, batch_size=50):
    
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        all_embeddings.extend(embed_texts(batch, task_type))

    return all_embeddings


@st.cache_data
def load_reviews():
    
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()

    
    df["embedding"] = embed_in_batches(
        df["comment"].astype(str).tolist(),
        task_type="RETRIEVAL_DOCUMENT",
    )

    df.to_parquet(CACHE_FILE, index=False)
    return df


def cosine_similarity(vec_a, vec_b):
    
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if denominator == 0:
        return 0

    return np.dot(vec_a, vec_b) / denominator


def find_similar_reviews(question, df):
    
    
    question_vector = embed_texts(
        [question],
        task_type="RETRIEVAL_QUERY",
    )[0]

    df = df.copy()

    df["score"] = df["embedding"].apply(
        lambda review_vector: cosine_similarity(question_vector, review_vector)
    )

    return df.nlargest(TOP_K, "score")


def ask_llm(question, top_reviews):
    
    context = ""

    for _, row in top_reviews.iterrows():
        context += (
            f"[Review ID: {row['review_id']} | "
            f"City: {row['city']} | Rating: {row['rating']}]\n"
            f"{row['comment']}\n\n"
        )

    prompt = f"""
You are an analytics assistant for a food delivery app.

Answer the user's question using ONLY the customer reviews below.
Be concise and answer in the same language as the question.
If the reviews do not contain enough information, say so clearly.
Do not invent facts.
At the end, include the review IDs used as sources.

Question:
{question}

Customer reviews:
{context}
"""

  
    interaction = client.interactions.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    return interaction.output_text


st.title("Chat with your Reviews")
st.caption(
    f"Searching {NEW_REVIEWS} reviews with Gemini, "
    f"then answering with {CHAT_MODEL}"
)

review_df = load_reviews()

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder="e.g. What are the most common complaints about delivery?",
)

if question:
    with st.spinner("Searching reviews and generating an answer..."):
        top_reviews = find_similar_reviews(question, review_df)
        answer = ask_llm(question, top_reviews)

    st.markdown("**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(
            top_reviews[["review_id", "city", "rating", "comment", "score"]],
            hide_index=True,
        )