import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# --------- Load once globally ---------
model = tf.keras.models.load_model("chatbot_model_transfer", compile=False)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

with open("intents.json") as file:
    data = json.load(file)

labels = [intent['tag'] for intent in data['intents']]
le = LabelEncoder()
le.fit(labels)

# --------- Function to get chatbot response ---------
def get_response(sentence):
    # Convert sentence to embedding
    vec = embed([sentence])
    
    # Predict intent
    pred = model.predict(vec)
    intent_tag = le.inverse_transform([np.argmax(pred)])[0]
    
    # Select a random response
    for intent in data['intents']:
        if intent['tag'] == intent_tag:
            response = np.random.choice(intent['responses'])
            break
    return response,intent_tag

# --------- Example usage ---------
#sentence = "i want to know abourtt the pgrkam website"
#response = get_chatbot_response(sentence)
#print("Chatbot:", response)
