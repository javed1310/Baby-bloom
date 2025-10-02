import streamlit as st
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()


def create_intent_classifier_chain(llm):
    """
    Creates a chain to classify the user's intent.
    """
    intent_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an intent classifier. Your job is to determine the user's intent based on their message.
        Respond with only one of three options: 'conversational_greeting', 'meta_query', or 'informational_query'.

        - 'conversational_greeting': For simple greetings like hello, hi, thanks, bye.
        - 'meta_query': For questions ABOUT the chatbot, its memory, or the conversation itself.
        - 'informational_query': For all other questions seeking medical or factual information.

        User message: {question}
        Intent:"""
    )
    return LLMChain(llm=llm, prompt=intent_prompt)

@st.cache_resource
def initialize_rag_system():
    """
    Initializes the RAG system by loading the pre-built vector database and setting up the chains.
    """
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    index_path = "faiss_direct_index"
    if not os.path.exists(index_path):
        st.error("Vector database not found! Please run the `build_database.py` script first to create it.")
        st.stop()
        
    with st.spinner("Loading knowledge base... This may take a moment."):
        db = FAISS.load_local(index_path, embedding_function, allow_dangerous_deserialization=True)

    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    
    base_retriever = db.as_retriever(search_kwargs={"k": 15})
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    
    custom_prompt_template = """
    **You are Baby Blooms, an AI Neonatal Information Assistant.** Your persona is empathetic, knowledgeable, and extremely focused on safety. Your expertise comes *exclusively* from the medical textbook excerpts provided in the **CONTEXT**.

    **Your primary goal is to structure your answer in a specific, multi-part format to ensure clarity and safety.**

    **CRITICAL INSTRUCTIONS:**
    1.  **Persona and Tone:** Maintain a calm, supportive, and professional tone. It is very important to sound reassuring.
    2.  **Grounding:** Your entire answer MUST be based *only* on the information within the **CONTEXT**. Do not use any external knowledge.
    3.  **No Mention of "Context":** NEVER mention the words 'context', 'provided text', or 'snippets'. Present the information as your own knowledge.
    4.  **Handling "I Don't Know":** If the **CONTEXT** does not contain the answer, you MUST respond *only* with: "I've carefully checked the medical textbooks, but I couldn't find specific information on this topic. For your child's safety and well-being, it's very important to consult with a pediatrician for guidance."
    5.  **MANDATORY ANSWER STRUCTURE:** For every informational query, you MUST structure your response in these exact three parts:

    
        * Start with a single empathetic sentence that acknowledges the user's concern.

        * Then, immediately explain the potential causes for the issue based on the provided context. You can use a sentence or bullet points.

    
        * Based on the context, describe the potential solutions, treatments, or management strategies that can be considered.
        * Frame this as general information from the textbooks. Use bullet points for clarity.

    
        * This is the most critical part. Conclude with a clear and firm recommendation to consult a healthcare professional.
        * Briefly explain *why* this is necessary (e.g., "for an accurate diagnosis and a personalized treatment plan for your baby").

    ---
    CONTEXT:
    {context}
    ---
    QUESTION:
    {question}
    ---
    
    """
    
    CUSTOM_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        verbose=False 
    )
    
    intent_classifier = create_intent_classifier_chain(llm)
    
    return {
        "qa_chain": conversation_chain, 
        "intent_classifier": intent_classifier
    }


st.title("👶 Baby Blooms: Your Newborn Care Assistant")
st.caption("Empathetic answers grounded in medical textbooks.")

if not os.getenv("GROQ_API_KEY") or not os.getenv("COHERE_API_KEY"):
    st.error("API keys for Groq and Cohere are missing from your .env file.")
    st.stop()

try:
    rag_system = initialize_rag_system()
    qa_chain = rag_system["qa_chain"]
    intent_classifier = rag_system["intent_classifier"]
except Exception as e:
    st.error(f"Failed to initialize the chatbot. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about newborn care..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Baby Blooms is thinking..."):
            intent_response = intent_classifier.invoke({"question": prompt})
            intent = intent_response.get('text', 'informational_query').strip().lower()

            if "conversational_greeting" in intent:
                result = "Hello! I'm Baby Blooms, an AI assistant ready to help with your questions about newborn health. How can I assist you today?"
            elif "meta_query" in intent:
                result = "I'm Baby Blooms, an AI assistant designed to answer questions based on information from medical textbooks. My memory helps me follow our current conversation, but I don't retain information from past sessions. How can I help with a newborn health question?"
            else: 
                response = qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.messages
                })
                result = response.get('answer', "I apologize, an error occurred while generating the response.")
            
            disclaimer = "\n\n---\n**Disclaimer:** I am an AI assistant. This information is for educational purposes only and is based on the provided textbooks. It is not a substitute for professional medical advice. Please consult a qualified healthcare provider or pediatrician for any health concerns."
            if "informational_query" in intent and "I couldn't find specific information" not in result:
                result += disclaimer

            st.markdown(result)
            
    st.session_state.messages.append({"role": "assistant", "content": result})