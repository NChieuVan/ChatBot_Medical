from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/chat", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("Question:", msg)

    response = rag_chain.invoke({"input": msg})
    contexts = response['context']

    # Chuyển page sang str để tránh TypeError
    sources = list({  # dùng set để loại trùng
    f"Page {str(item.metadata.get('page','unknown'))} - {item.metadata.get('source','unknown')}"
    for item in contexts
    })
    source_text = "\n".join(sources)

    # print("Response:", response["answer"])
    # print("Sources:", source_text)

    return f"{response['answer']}\n\nSources:\n{source_text}"




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)