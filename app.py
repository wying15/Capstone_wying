import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import openai
from PIL import Image
import random
import os
import json
import requests
import base64
from io import BytesIO
from streamlit_extras.let_it_rain import rain


st.set_page_config(page_title="Chat with Rainbow Riley", page_icon="👩🏻‍🏫", layout="centered", initial_sidebar_state="auto", menu_items=None)

#Context

# Set OpenAI API key
openai.api_key = st.secrets.openai.api_key


# URL of the image you want to display
image_url = "https://raw.githubusercontent.com/wying15/Capstone_wying/40d4335de8e96f83ff64165eda51b68206f88712/Streamlit/BannerRainbow.png"

# Display the image in Streamlit using HTML and CSS
st.markdown(f"""
<style>
.shifted-image {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 500px;
    height: 133px; /* Adjust the height as needed */
    margin-top: 50px; /* Adjust this value to bring the image closer to the top */
}}
</style>
<img class="shifted-image" src="{image_url}" />
""", unsafe_allow_html=True)

# URL of the GIF you want to display
gif_url = "https://raw.githubusercontent.com/wying15/Capstone_wying/main/Streamlit/giphy.webp"

# Fetch the GIF from the URL
response = requests.get(gif_url)

# Convert the GIF to base64
gif_base64 = base64.b64encode(response.content).decode('utf-8')

# Display the GIF in Streamlit using base64-encoded string, center it, and adjust its height and position
st.markdown(f"""
<style>
.center {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 500px;
    height: 463px; /* Adjust the height as needed */
    margin-top: 10px; /* Push the GIF down */
}}
.caption {{
    text-align: center;
    margin-top: 20px; /* Adjust the margin-top as needed */
}}
</style>
<img class="center" src="data:image/gif;base64,{gif_base64}" />
<div class="caption">How Can I Help You Today?</div>
""", unsafe_allow_html=True)


# Display centered text
#st.markdown("<p style='text-align: center;'>Welcome to Rainbow Riley!</p>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Balancing the 5 flavour elements... Please wait while we create a harmonious culinary experience! This may take 1-2 minutes."):
        
        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(persist_dir="main/data/index.vecstore")

        # Load the index
        index = load_index_from_storage(storage_context)

        # Load the finetuned model 
        ft_model_name = "ft:gpt-3.5-turbo-1106:personal:capstone-exp-3:9vgnLOrh"
        ft_context = ServiceContext.from_defaults(llm=OpenAI(model=ft_model_name, temperature=0.3), 
        context_window=2048, 
        
        system_prompt="""
       Craft a series of questions that customers might ask about what fruits and vegetables to add to their diet to achieve the optimal nutrition. The optimal nutrition includes at least 1 item from the Red column, Yellow & Orange column, White, Tan & Brown column, Green column and Blue & Purple column of the same row of the data file. Check if the user input has at least one item from each of these columns and then provide the full row of values that are closest to the user input which must include the full recipe in the corresponding Recipes column.
        """
        )           
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask Me Rainbow Riley Questions Relating to fun ingredients and yummy recipes.😊"}
    ]

if prompt := st.chat_input("Ask Me Rainbow Riley Questions Relating to fun ingredients and yummy recipes."):
    # Save the original user question to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.new_question = True

    # Create a detailed prompt for the chat engine
    chat_history = ' '.join([message["content"] for message in st.session_state.messages])
    detailed_prompt = f"{chat_history} {prompt}"

if "new_question" in st.session_state.keys() and st.session_state.new_question:
   for message in st.session_state.messages: # Display the prior chat messages
       with st.chat_message(message["role"]):
           st.write(message["content"])
   st.session_state.new_question = False # Reset new_question to False

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
   with st.chat_message("assistant"):
       with st.spinner("Calculating..."):
           response = chat_engine.chat(detailed_prompt)
           st.write(response.response)
           # Append the assistant's detailed response to the chat history
           st.session_state.messages.append({"role": "assistant", "content": response.response})
