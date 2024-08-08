import json
import boto3
import streamlit as st
import dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

st.title("Demo")


llm = Ollama(model="llama3")

history = StreamlitChatMessageHistory(key="chat_messages")
if len(history.messages) == 0:
    history.add_ai_message("How can I help you?")

prompt = ChatPromptTemplate.from_template("You answer always in korean. Answer the question. <Question>: {input}")

def parse_stream(stream):
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            message = json.loads(chunk.get("bytes").decode())
            if message['type'] == "content_block_delta":
                yield message['delta']['text'] or ""
            elif message['type'] == "message_stop":
                return "\n"

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

chain = prompt | llm

if human_input := st.chat_input(key="input"):
    st.chat_message("human").write(human_input)
    history.add_user_message(human_input)

    response = chain.invoke({"input": human_input})

    st.chat_message("ai").write(response)
    history.add_ai_message(response)