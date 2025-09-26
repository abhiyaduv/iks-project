# Install dependencies first:
# pip install streamlit langchain-community faiss-cpu sentence-transformers transformers requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st
import datetime
import requests

# ---------------- FAQ Data ----------------
faq_data = [
    {"question": "What is the best time to visit Mumbai?",
     "answer": "The best time to visit Mumbai is from November to February when the weather is cool and pleasant."},

    {"question": "Where are the popular tourist spots in Mumbai?",
     "answer": "Some popular spots include Gateway of India, Marine Drive, Juhu Beach, Chhatrapati Shivaji Terminus, and Elephanta Caves."},

    {"question": "Tell me about public transport in Mumbai",
     "answer": "Mumbai has local trains, buses, metro, auto-rickshaws, and taxis. Trains are the fastest way to travel across the city."},

    {"question": "How can I check the weather in Mumbai?",
     "answer": "You can check live weather using apps or websites like OpenWeatherMap or Google Weather."},

    {"question": "What is Mumbai famous for?",
     "answer": "Mumbai is famous for Bollywood, beaches, street food, historical landmarks, and as the financial capital of India."},

    {"question": "Tell me about local cuisine in Mumbai",
     "answer": "Famous foods include Vada Pav, Pav Bhaji, Bhel Puri, Misal Pav, and Bombay Sandwich."},

    {"question": "Where is the airport located?",
     "answer": "Chhatrapati Shivaji Maharaj International Airport is located in Sahar, Andheri East, Mumbai."}
]

# ---------------- Prepare Documents ----------------
docs = [Document(page_content=f["answer"], metadata={"question": f["question"]}) for f in faq_data]

# Embedding and Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# QA Model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Mumbai FAQ Chatbot", page_icon="🏙️", layout="centered")

st.title("🏙️ Mumbai FAQ Chatbot")
st.markdown("Ask anything about **tourist spots, transport, cuisine, weather, or general info** about Mumbai.")

# Sidebar Quick Links
st.sidebar.title("📌 Quick FAQs")
quick_links = [
    "Best time to visit",
    "Tourist spots",
    "Public transport",
    "Local cuisine",
    "Airport info",
    "Weather info"
]
for item in quick_links:
    st.sidebar.markdown(f"- {item}")

# Main Input
query = st.text_input("💬 Type your question here:")

if st.button("Get Answer") and query:
    # Handle special commands
    if "date" in query.lower():
        st.success(f"📅 Today's date: {datetime.datetime.now().strftime('%d-%m-%Y')}")
    elif "time" in query.lower():
        st.success(f"⏰ Current time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    elif "weather" in query.lower():
        try:
            WEATHER_API_KEY = "your_openweathermap_api_key"  # Replace with your OpenWeatherMap API key
            CITY = "Mumbai"
            url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={WEATHER_API_KEY}&units=metric"
            data = requests.get(url).json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            st.success(f"🌤️ Weather in {CITY}: {desc}, {temp}°C")
        except:
            st.warning("Sorry, couldn't fetch the weather.")
    else:
        result = qa_chain(query)
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        if source_docs:
            matched_question = source_docs[0].metadata["question"]
            st.success(f"**Answer:** {answer}")
            st.info(f"📌 Based on: *{matched_question}*")
        else:
            st.warning("Sorry, I couldn't find an answer. Try rephrasing your question.")
