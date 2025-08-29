from openai import OpenAI
import os

def main():
    # Create client (reads OPENAI_API_KEY from env automatically)
    client = OpenAI(api_key="youtr open ai keys")

    # Simple prompt test
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can also try "gpt-4o" or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write me a haiku about AI and the future."}
        ],
        max_tokens=100
    )

    # Print response text
    print("\n=== OpenAI API Test ===\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
