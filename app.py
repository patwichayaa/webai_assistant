import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import openai
import tiktoken
from io import BytesIO
import base64


# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to count tokens
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# Function to get text embeddings
def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        engine='text-embedding-ada-002'
    )
    return response['data'][0]['embedding']

# Function to refine clusters using OpenAI ChatGPT
def chatgpt_refine_clusters(texts, embeddings, prompt):
    prompt_message = (
        "Here is a list of texts."
        "Please suggest main groups/categories for these texts based on their content similarity.\n\n"
    )
    for i, text in enumerate(texts):
        prompt_message += f"Text {i+1}: {text}\n\n"
    
    prompt_message += f"\nUser Prompt: {prompt}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_message}
        ],
        temperature=0.3,
        max_tokens=4000
    )
    refined_labels = response['choices'][0]['message']['content'].strip()
    return refined_labels

st.set_page_config(layout="wide")

# Function to inject custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to create the UI for text clustering
def create_text_clustering_ui():
    # "Back to Home" button on the top right
    if st.button("Back to Home", key="back_to_home"):
        st.session_state.page = 'Home'

    
    st.header("Text Clustering")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    # Create side bar with styled expander
    with st.sidebar:
        
        # Input field for API key
        api_key = st.text_input("Enter OpenAI API key", type="password")
        
        with st.expander("What is Text Clustering"):
            st.markdown('<div class="expander-content">Function for the automatic extraction of the main topics of a text or function through the auto-classification of the keyphrases contained in the text. To obtain the summary of topics and related keyphrases</div>', unsafe_allow_html=True)
        with st.expander("How to use"):
            st.markdown('<div class="expander-content">Step 1 : Select option <br>Step2: Import excel file <br>Step 3 : Click button cluster(read only first column)</div>', unsafe_allow_html=True)
        option = st.selectbox("Option", ("Auto Clustering", "Have Condition", "One to Many"))
        st.write("You selected:", option)
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        st.write("Dataframe:")
        st.dataframe(df)
        
        if not df.empty:
            all_texts = df.iloc[:, 0].astype(str).tolist()  # Convert to string and get first column
            all_texts_combined = " ".join(all_texts)  # Combine all texts into one string
            token_count = count_tokens(all_texts_combined)
            st.write(f"Token count for all texts in the first column: {token_count}")
        
        return df, option
    
    return None, option

def parse_clusters(refined_labels, num_texts):
    clusters = {}
    lines = refined_labels.split("\n")
    current_cluster = None

    for line in lines:
        if line.strip() == "":
            continue
        if ':' in line:
            current_cluster, texts = line.split(':', 1)
            current_cluster = current_cluster.strip()
            texts = texts.strip()
            if texts:
                # Extract text indices
                text_indices = texts.split(", ")
                clusters[current_cluster] = text_indices
        elif current_cluster and "Text" in line:
            # Handle case where clusters continue on the next lines
            texts = line.split(":")[1].strip()
            text_indices = texts.split(", ")
            if current_cluster in clusters:
                clusters[current_cluster].extend(text_indices)
            else:
                clusters[current_cluster] = text_indices

    # Create a mapping of text index to cluster
    text_to_cluster = {}
    for cluster, text_indices in clusters.items():
        for text in text_indices:
            try:
                index = int(text.split(" ")[1]) - 1
                if 0 <= index < num_texts:
                    text_to_cluster[index] = cluster
            except (IndexError, ValueError):
                # Handle the error gracefully
                print(f"Skipping invalid text: {text}")

    return text_to_cluster



# Main function
def main():
    # Set menu options based on page state
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    
    # Load the custom CSS file
    local_css("style.css")

    # Use markdown for custom styling
    if st.session_state.page == 'Home':
        selected = option_menu(
            menu_title=None,  # required
            options=["Home"],  # required
            icons=["house"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        
        # Load the image and convert to base64
        def get_image_base64(file_path):
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()

        image_base64 = get_image_base64("nebula.png")

        # Define custom CSS to style the text and image
        st.write(f"""
            <style>
                .container {{
                    position: relative;
                    text-align: center;
                    color: black;
                }}
                .text {{
                    font-family: 'Poppins', sans-serif;
                    font-size: 40px;
                    position: center;
                    z-index: 1;
                }}
                .background-image {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-55%, -115%);
                    width: 250px; /* Adjust the size as needed */
                    height: auto;
                    z-index: 0;
                }}
            </style>
            <div class="container">
                <img src="data:image/png;base64,{image_base64}" class="background-image">
                <h1 class="text poppins-bold" style='color: #ff4b4b; font-size: 60px; margin-top: 40px; text-align: center; line-height: 1.2;'>
                    AI ASSISTANT<br>
                    <span style='color: black; font-size: 40px; display: block; margin-bottom: -100px;'>Your personal AI</span><br>
                     <span style='color: white; font-size: 25px;'>am</span><span style='color: black; font-size: 25px;'>Powered by City Innovation Team</span>
                    <span class='text'></span>
                </h1>

            </div>
        """, unsafe_allow_html=True)
        

        st.write("""
        <div style='text-align: center;'>
            <p style='font-family: Poppins, sans-serif; font-size: 15px;'>
                Save time on research, documents management and content generation. <br> 
                AI Assistant uses that knowledge to give you superpowers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <h1 class="poppins-bold" style='color: black; font-size: 40px;'>Our Features</h1>
        </div>
        """, unsafe_allow_html=True)
        st.write("""
        <div style='text-align: center;'>
            <p style='font-family: Poppins; font-size: 15px;'>Increase productivity, streamline tasks and keep information secure with your own AI Assistant.
            </p>
        </div>    
           """, unsafe_allow_html=True)

        # Define a custom CSS for centering
        st.markdown("""
            <style>
                .center-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 40vh; /* Full height of the viewport */
                }
                .center-content {
                    width: 450px;
                    height: 280px;
                    background: linear-gradient(50deg, #fd9277, #ffcc99);
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                    border-radius: 15px;
                    text-align: center;
                    line-height: 1.0; 
                }
            </style>
        """, unsafe_allow_html=True)

        # Container to center content vertically and horizontally
        with st.container():
            # Create three columns with equal width
            col1, col2, col3 = st.columns(3)

            # First column
            with col1:
                st.markdown("""
                    <div class="center-container" style="display: flex; align-items: center;">
                        <div class="center-content" style="flex: 1;">
                            <p style='font-family: Poppins; text-align: left; margin-left: 30px; font-size: 20px;'>
                                <span style='font-weight: bold; border-bottom: 2px solid white; margin-top: 40px; margin-bottom: 8px; display: inline-block;'>AI Text Clustering</span><br>
                                <span style="color: white; font-size: 12px; text-align: left; margin-top: 10px; line-height: 2.0 !important;">
                                    AI Text Clustering groups similar documents or sentences by content and meaning, enabling efficient categorization and analysis. It eliminates the need to scroll through a long text <br> label, allowing you to focus on the most relevant information <br> and save time.
                                </span>
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                if st.button("Go To Text Clustering", key='text_clustering_button'):
                    st.session_state.page = 'Text Clustering'

            # Second column
            with col2:
                
                st.markdown("""
                    <div class="center-container" style="display: flex; align-items: center;">
                        <div class="center-content" style="flex: 1;">
                            <p style='font-family: Poppins; text-align: left; margin-left: 30px; font-size: 20px;'>
                                <span style='font-weight: bold; border-bottom: 2px solid white; margin-top: -15px; margin-bottom: 8px; display: inline-block;'>AI Chatbot (In the future)</span><br>
                                <span style="color: white; font-size: 12px; text-align: left; margin-top: 10px; line-height: 2.0 !important;">
                                    that will help you generate task and notes, summarize text, write content, brainstorm and mind map ideas, and >more. All this is possible inside the project chat box.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
                
            # Third column
            with col3:
                
                st.markdown("""
                    <div class="center-container" style="display: flex; align-items: center;">
                        <div class="center-content" style="flex: 1;">
                            <p style='font-family: Poppins; text-align: left; margin-left: 30px; font-size: 20px;'>
                                <span style='font-weight: bold; border-bottom: 2px solid white; margin-top: 40px; margin-bottom: 8px; display: inline-block;'>AI Image Clustering (In the future)</span><br>
                                <span style="color: white; font-size: 12px; text-align: left; margin-top: 10px; line-height: 2.0 !important;">
                                      AI image clustering is a technique used to automatically group similar images into clusters or categories based on their visual content. Algorithms to analyze and identify patterns within images, helping to organize large collections of visual data without manual labeling.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

    elif st.session_state.page == 'Text Clustering':
        df, option = create_text_clustering_ui()
        
        if option == "Auto Clustering":
                prompt = (
        "Here is a list of texts."
        "Please suggest main groups/categories for these texts based on their content similarity.\n\n"
    )
       
        elif option == "Have Condition":
            num_clusters = st.slider("Select the number of clusters:", min_value=1, max_value=20, value=3, step=1)
            conditions = st.text_area("Enter conditions for clustering:", "Suggest categories based on the content.")
            prompt = f"Here is a list of texts.Please suggest {num_clusters} categories for these texts based on the following conditions:\n{conditions} label categories in each of text"
        else:  # option == "Many"
            prompt = "Suggest one or more categories for each text based on its content. If a text fits multiple categories, list them all, one text in one per line"

        if df is not None:
            texts = df.iloc[:, 0].tolist()  # Process only the first column
            embeddings = [get_embeddings(text) for text in texts]
            X = np.array(embeddings)

            if st.button("Clusters"):
                refined_labels = chatgpt_refine_clusters(texts, embeddings, prompt)  # Pass embeddings here

                if option == "Auto Clustering":
                    # Process for "General" option
                    refined_labels_list = refined_labels.split("\n")
                    text_to_cluster = parse_clusters(refined_labels, len(texts))
                    
                    # Initialize list to store refined labels per text
                    refined_labels_per_text = [''] * len(texts)

                    # Assign clusters to texts based on the parsed data
                    for idx, cluster in text_to_cluster.items():
                        if 0 <= idx < len(refined_labels_per_text):
                            refined_labels_per_text[idx] = cluster

                    # Add refined labels to DataFrame
                    df['Cluster'] = refined_labels_per_text
                    st.write(refined_labels)
                    st.write("Cluster Labeling:")
                    st.dataframe(df)

                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Refined_Labels')
                        writer.close()  # Correct method to close the writer
                        processed_data = output.getvalue()
                    
                    st.download_button(
                        label="Download Excel File",
                        data=processed_data,
                        file_name='text_clustering.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                elif option == "Have Condition":
                    # Display refined labels directly for "Have Condition" option
                    st.write("Cluster Labeling:")
                    st.write(refined_labels)

                else:  # option == "Many"
                    # For "Many" option: Display refined labels directly
                    st.write("Cluster Labeling:")

                    # Convert the refined_labels to a list of texts and corresponding labels
                    refined_labels_list = refined_labels.split("\n")

                    # Create a formatted HTML string to display each text with its corresponding label on a new line
                    formatted_labels = "<br>".join([f"{label}" for i, label in enumerate(refined_labels_list)])

                    # Display the formatted labels with HTML
                    st.markdown(f"<div style='font-family: Poppins, sans-serif; font-size: 15px; white-space: pre-line;'>{formatted_labels}</div>", unsafe_allow_html=True)

                    # Add refined labels to DataFrame
                    df['Cluster'] = refined_labels_list

                    st.write("Cluster Labeling:")
                    st.dataframe(df)

                    # Create a BytesIO object to save the Excel file
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Refined_Labels')
                        writer.close()  # Correct method to close the writer
                        processed_data = output.getvalue()

                    # Provide the download button
                    st.download_button(
                        label="Download Excel File",
                        data=processed_data,
                        file_name='text_clustering.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == "__main__":
    main()
