#!/usr/bin/env python3
from anthropic import AnthropicVertex
import os
import streamlit as st
### Google Vertex AI imports
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Tool,
    grounding
)
from vertexai.preview.vision_models import ImageGenerationModel

#Basic environment settings
PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = 'us-east5'
VS_REGION = "us-central1"
DATASTORE_ID = "cloud-arch-center-datastore_1729197031195"


vertexai.init(project=PROJECT_ID, location=LOCATION)


# --- Anthropic Setup ---
client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

@st.cache_resource
def load_models():
    """Load Gemini 1.5 Flash, Pro models, and Anthropic Claude."""
    return (
        GenerativeModel("gemini-1.5-flash"),
        GenerativeModel("gemini-1.5-pro"),
        "claude-3-5-sonnet@20240620",  # Add Claude model
    )

def get_gemini_response(
    model: GenerativeModel,
    tool: Tool,
    contents: str | list,
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=2048
    ),
    stream: bool = True,
    rag: bool = False
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

#>>> response.candidates[0].grounding_metadata.grounding_chunks
    if (rag):
        responses = model.generate_content(
            contents,
            tools=[tool],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
        )
    else:
        responses = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
        )

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def get_anthropic_response(
    model: str, 
    prompt: str, 
    max_tokens: int = 1024
) -> str:
    """Generate a response from the Anthropic Claude model."""
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return message.content[0].text


def get_model_name(model):
    """Get Model Name (Gemini or Claude)"""
    if isinstance(model, GenerativeModel):
        model_name = model._model_name.replace(  # pylint: disable=protected-access
            "publishers/google/models/", ""
        )
        return f"`{model_name}`"
    else:
        return f"`{model}`"  # For Claude


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def prepare_prompt(query: str, length: str, rag: bool) -> str:
    """Prepare prompt using template and RAG context if requested"""
    if rag:
        prompt = f"""Provide a {length} and answer to {query} \n
        Response should be step by step and based on Google Architecture Centre Documentation. 
        Include URLs to relevant articles in the Google Architecture Centre Documentation.
        If the length of response is "short" then make sure to have 2 paragraph or else if it is "long" then respond with between 4 to 6 paragraphs maximum.
        Include Terraform if applicable, always generate an illustrative architecture diagram
        Important point is that each response should be generated based on the premise given above.
         """
    else:
        prompt = f"""Provide a {length} and answer to {query} \n
        Response should be step by step and based on Google Architecture Centre Documentation. 
        If the length of response is "short" then make sure to have 2 paragraph or else if it is "long" then respond with between 4 to 6 paragraphs maximum.
        Include Terraform if applicable, always generate an illustrative architecture diagram
        Important point is that each response should be generated based on the premise given above.
         """

    return prompt

st.header("Not so awesome design generator", divider="rainbow")
gemini_15_flash, gemini_15_pro, claude_model = load_models()
groundtool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

# --- Chat with me ---
st.subheader("What do you want to build")

selected_model = st.radio(
    "Select Model:",
    [gemini_15_flash, gemini_15_pro, claude_model],
    format_func=get_model_name,
    key="selected_model_story",
    horizontal=True,
)

# Story premise
question_body = st.text_input(
    "Hi there how can I help you: \n\n",
)

length_of_response = st.radio(
    "Select the length of the response: \n\n",
    ["Short", "Long"],
    key="length_of_story",
    horizontal=True,
)

if length_of_response == "Short":
    max_output_tokens = 2048
else:
    max_output_tokens = 8192

enable_rag = st.radio(
    "Use RAG: \n\n",
    ["No-RAG", "RAG"],
    horizontal=True,
)

prompt = prepare_prompt(query=question_body, length=length_of_response, rag=enable_rag)

config = GenerationConfig(
    max_output_tokens=max_output_tokens
)

generate_t2t = st.button("Answer here", key="generate_t2t")
if generate_t2t and question_body:
    with st.spinner(
        f"Let me think {get_model_name(selected_model)} ..."
    ):
        # Generate response (Gemini or Claude)
        if isinstance(selected_model, GenerativeModel):
            response = get_gemini_response(
                selected_model,
                groundtool,
                prompt,
                generation_config=config,
                rag=enable_rag
            )
        else:  # Anthropic
            response = get_anthropic_response(
                selected_model,
                prompt,
                max_tokens=max_output_tokens
            )

        if response:
            st.write("My Thoughts:")
            st.write(response)

        # Generate Image with Imagen
        with st.spinner("Generating image..."):
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
            images = model.generate_images(
                prompt=f"""Provide a sketch simple flow style architectural diagram, with steps based on {response} """,
                number_of_images=1,
            )
            if images:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    images[0].save(location=temp_file.name)
                    st.image(temp_file.name)

