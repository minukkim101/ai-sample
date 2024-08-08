import json
import boto3
import streamlit as st
import dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

dotenv.load_dotenv()

st.title("Demo")


client = boto3.client("bedrock-runtime")

# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
#model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"

history = StreamlitChatMessageHistory(key="chat_messages")
if len(history.messages) == 0:
    history.add_ai_message("How can I help you?")


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

if human_input := st.chat_input(key="input"):

    st.chat_message("human").write(human_input)
    history.add_user_message(human_input)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": "You answer always in korean.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": human_input}],
            }
        ],
    })

    streaming_response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=body,
    )

    stream = streaming_response.get("body")

    ai_response = ""
    ai_message_placeholder = st.empty()

    for chunk in parse_stream(stream):
        ai_response += chunk
        ai_message_placeholder.markdown(f"**AI:** {ai_response}")

    # Add the final AI response to the history
    history.add_ai_message(ai_response)