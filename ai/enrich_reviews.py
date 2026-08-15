import os
import snowflake.connector
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash"

SAMPLE_N = 5
TOPICS = ["food quality", "delivery", "pricing", "service", "packaging", "other"]

client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = f"""
You classify customer reviews for a food delivery app.

For the review you are given, return:
- sentiment_label: positive, negative, or neutral
- sentiment_score: a number between -1.0 and 1.0
- topic: one of {TOPICS}
- key_issue: a short phrase of 6 words or less that describes the main issue
  in the review, if any. If there is no issue, return null.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment_label": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
        },
        "sentiment_score": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "topic": {
            "type": "string",
            "enum": TOPICS,
        },
        "key_issue": {
            "type": "string",
            "nullable": True,
        },
    },
    "required": [
        "sentiment_label",
        "sentiment_score",
        "topic",
        "key_issue",
    ],
}

def get_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

def create_output_table(cursor):
    cursor.execute("CREATE SCHEMA IF NOT EXISTS ZOMATO.AI")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ZOMATO.AI.REVIEW_ENRICHED (
            REVIEW_ID STRING,
            SENTIMENT_LABEL STRING,
            SENTIMENT_SCORE FLOAT,
            TOPIC STRING,
            KEY_ISSUE STRING,
            MODEL STRING,
            ENRICHED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)

def get_reviews_to_enrich(cursor):
    cursor.execute(f"""
        SELECT REVIEW_ID, COMMENT
        FROM ZOMATO.RAW.REVIEWS
        WHERE REVIEW_ID NOT IN (
            SELECT REVIEW_ID FROM ZOMATO.AI.REVIEW_ENRICHED
        )
        LIMIT {SAMPLE_N}
    """)
    return cursor.fetchall()

def classify_review(comment):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"{SYSTEM_PROMPT}\n\nReview:\n{comment}",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    return response.parsed

def save_results(cursor, results):
    if not results:
        return

    print(f"Saving {len(results)} enriched reviews to Snowflake...")
    cursor.executemany(
        """
        INSERT INTO ZOMATO.AI.REVIEW_ENRICHED
            (review_id, sentiment_label, sentiment_score, topic, key_issue, model)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        results,
    )

def main():
    conn = get_connection()
    cursor = conn.cursor()

    try:
        create_output_table(cursor)
        reviews = get_reviews_to_enrich(cursor)

        if not reviews:
            print("No new reviews to enrich.")
            return

        print(f"Enriching {len(reviews)} reviews...")
        results = []

        for review_id, comment in reviews:
            print(f"Classifying review {review_id}: {comment}")

            try:
                labels = classify_review(comment)
                print(f"Labels for review {review_id}: {labels}")

                results.append((
                    review_id,
                    labels["sentiment_label"],
                    labels["sentiment_score"],
                    labels["topic"],
                    labels["key_issue"],
                    MODEL,
                ))
            except Exception as e:
                print(f"Error classifying review {review_id}: {e}")

        save_results(cursor, results)
        conn.commit()
        print(f"Saved {len(results)} enriched reviews to Snowflake.")

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()