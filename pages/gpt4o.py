import streamlit as st
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()
# Initialize API client with credentials


def gpt40_chat(user_query):
    """Send a query to the GPT-4.0 model and return the response."""
    try:
        response = client.chat.completion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": user_query},
            ],
            temperature=0.4,
            max_tokens=1000,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

def encode_image_to_base64(image_file):
    """Encode an image file to base64."""
    try:
        return base64.b64encode(image_file.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Failed to encode image: {e}")
        return None

def analyze_image(image_file):
    """Analyze a single uploaded image and generate a description."""
    try:
        base64_image = encode_image_to_base64(image_file)
        if base64_image:
            user_query = f"Look at this image and explain it: data:image/jpeg;base64,{base64_image}"
            description = gpt40_chat(user_query)
            return description
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit App
def run_app():
    st.title('Image Analysis App')
    
    uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            description = analyze_image(uploaded_file)
            if description:
                st.write(f"Analysis for {uploaded_file.name}: {description}")


if __name__ == "__main__":
    run_app()

