from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models.bedrock import BedrockChat
# from langchain.chat_models.anthropic import ChatAnthropic
import streamlit as st
from langchain_aws import ChatBedrock

# Reference: https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history
st.title("MessageHistory")

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs=dict(temperature=0),
    streaming=True
    # other params...
)

history = StreamlitChatMessageHistory(key='chat_messages')
if len(history.messages) == 0:
    history.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

system_prompt = "You answer always in korean."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            """
            Current conversation:
            <conversation_history>
            {history}
            </conversation_history>

            Here is the human's next reply:
            <human_reply>
            {human_input}
            </human_reply>
            """,
        ),
    ]
)

chain = prompt | llm | StrOutputParser()


for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if human_input := st.chat_input(key="input"):
    st.chat_message("human").write(human_input)
    history.add_user_message(human_input)
    response = chain.invoke({"history": history, "human_input": human_input})
    history.add_ai_message(response)
    st.chat_message("ai").write(response)

