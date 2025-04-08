# chatbot.py

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# Sample chatbot responses
responses = [
    "Hello! How can I help you?",
    "I'm a chatbot created using Python.",
    "I'm doing great, thanks for asking!",
    "Goodbye! Have a nice day!",
    "I can answer simple questions."
]

# Tokenizer for NLP
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Generate a chatbot response
def chatbot_response(user_input):
    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english')
    all_texts = responses + [user_input]
    vectors = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    index = similarity.argmax()
    return responses[index]

# Main chat loop
if __name__ == "__main__":
    print("Chatbot: Hello! Type 'exit' to leave.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user_input))

