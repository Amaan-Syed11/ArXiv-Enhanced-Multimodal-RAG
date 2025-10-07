import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from dotenv import load_dotenv
import os
from PIL import Image
import io
import fitz  # PyMuPDF
import logging
import base64
from openai import OpenAI
import time
import tenacity
import arxiv
from langchain_core.messages import HumanMessage, AIMessage
import torch
import numpy as np

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the embedding model with OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# Initialize the LLM for conversational retrieval and key term extraction
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0.7
)

# Initialize OpenAI client for image summarization
client = OpenAI(api_key=OPENAI_API_KEY)



@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.info(f"Retrying in {retry_state.next_action.sleep} seconds...")
)
def summarize_images_in_batch(images, client, resize=False, max_size=(800, 800), quality=85):
    """Summarize a batch of images using GPT-4o in a single API call."""
    try:
        # Convert images to base64
        base64_images = []
        for image in images:
            if resize:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(img_str)

        # Prepare the message for GPT-4o
        user_content = [{"type": "text", "text": "Please describe each image in detail, one description per image, separated by '---'."}]
        for img_str in base64_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=150 * len(images),  # Allocate tokens for each image
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise descriptions of images."},
                {"role": "user", "content": user_content}
            ]
        )

        # Parse the response
        summary_text = response.choices[0].message.content
        summaries = summary_text.split("---")
        summaries = [s.strip() for s in summaries]

        # Ensure the number of summaries matches the number of images
        if len(summaries) != len(images):
            logging.warning(f"Mismatch in number of summaries ({len(summaries)}) and images ({len(images)}). Padding with error messages.")
            summaries.extend(["Unable to summarize this image."] * (len(images) - len(summaries)))
        summaries = summaries[:len(images)]  # Truncate if too many summaries

        logging.info(f"Successfully generated summaries for {len(summaries)} images in batch.")
        return summaries
    except Exception as e:
        logging.error(f"Error summarizing image batch: {str(e)}")
        return ["Unable to summarize this image."] * len(images)

# REPLACE your existing batch_summarize_images function with this one

def batch_summarize_images(images, client, batch_size=50):
    """Process images in large batches in parallel with error recovery and a progress bar."""
    if not images:
        return []

    summaries = []
    total_images = len(images)
    
    # Create a list of all batches to be processed
    batches = [images[i:i + batch_size] for i in range(0, total_images, batch_size)]
    
    # Initialize a list to store results in the correct order
    results = [None] * len(batches)
    
    progress = st.progress(0)
    processed_count = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all batches to the executor and store the future objects
        # We map each future to its original index to maintain order
        future_to_index = {
            executor.submit(summarize_images_in_batch, batch, client, resize=True): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            batch_size_of_completed_future = len(batches[index])
            try:
                # Get the result from the future
                batch_summaries = future.result()
                results[index] = batch_summaries
            except Exception as e:
                logging.error(f"Error processing batch at index {index}: {str(e)}")
                # Populate with error messages if a whole batch fails
                results[index] = ["Unable to summarize this image."] * batch_size_of_completed_future
            
            # Update progress bar
            processed_count += batch_size_of_completed_future
            progress.progress(processed_count / total_images)

    # Flatten the list of lists (results) into a single list of summaries
    summaries = [summary for batch_result in results for summary in batch_result]
    
    logging.info(f"Successfully processed {len(summaries)} image summaries in parallel.")
    return summaries

def extract_images_from_pdf(pdf_docs):
    """Extract all images from uploaded PDF documents in parallel using PyMuPDF."""
    images = []

    def process_pdf_for_images(pdf):
        """Helper function to extract images from a single PDF."""
        pdf_images = []
        try:
            logging.info(f"Processing file: {pdf.name}, Size: {len(pdf.getvalue())} bytes")
            pdf.seek(0)
            pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
            for page_index in range(len(pdf_file)):
                page = pdf_file[page_index]
                image_list = page.get_images(full=True)
                logging.info(f"Found {len(image_list)} images on page {page_index + 1}")

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image["image"]

                    try:
                        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        pdf_images.append(img_pil)
                        logging.info(f"Extracted image {img_index + 1} from page {page_index + 1}")
                    except Exception as e:
                        logging.error(f"Failed to process image {img_index + 1} on page {page_index + 1}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing PDF {pdf.name}: {str(e)}")
        finally:
            pdf.seek(0)
        return pdf_images

    # Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_pdf_for_images, pdf) for pdf in pdf_docs]
        for future in as_completed(futures):
            pdf_images = future.result()
            images.extend(pdf_images)

    logging.info(f"Total images extracted: {len(images)}")
    return images

def get_pdf_text_and_tables(pdf_docs):
    """Extract text and tables from uploaded PDF documents in parallel."""
    text = ""
    tables = []

    def process_pdf(pdf):
        """Helper function to process a single PDF."""
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                pdf_text = ""
                pdf_tables = []
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
                    pdf_tables.extend(page.extract_tables())
                return pdf_text, pdf_tables
        except Exception as e:
            logging.error(f"Error extracting text and tables from PDF {pdf.name}: {e}")
            return "", []

    # Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_pdf, pdf) for pdf in pdf_docs]
        for future in as_completed(futures):
            pdf_text, pdf_tables = future.result()
            text += pdf_text + "\n"
            tables.extend(pdf_tables)

    if not text.strip() and not tables:
        st.error("Failed to extract text and tables from the PDFs. Please check the file format.")

    return text, tables

# REPLACE your old setup_conversation_retriever function with this new one
def setup_conversation_retriever(combined_text, tables, image_summaries, embeddings, llm):
    """
    Sets up a modern, fully-streaming RAG pipeline using LCEL.
    """
    try:
        # --- The Retriever setup remains the same ---
        text_docs = [Document(page_content=combined_text)]
        table_docs = [Document(page_content="Table: " + "\n".join(["\t".join(map(str, row)) for row in table])) for table in tables]
        image_docs = [Document(page_content=f"Summary of an image: {s}") for s in image_summaries]
        all_docs = text_docs + table_docs + image_docs

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        vectorstore = FAISS.from_texts(["init"], embeddings)
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        logging.info("Adding documents to the ParentDocumentRetriever...")
        retriever.add_documents(all_docs, ids=None)
        logging.info("Documents added successfully.")

        # --- New LCEL Chain Implementation ---

        # 1. Condense Question Prompt: Combines chat history and new question
        condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        
        Chat History:
        {chat_history}
        
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(condense_question_template)

        # 2. Answer Prompt: The main prompt for answering the question with context
        answer_template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        ANSWER_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", answer_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}"),
            ]
        )
        
        # 3. Helper function to format retrieved documents
        def _format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 4. Build the final chain
        # This function handles the logic: if there's no history, use the question as is.
        # Otherwise, generate a new standalone question.
        def _get_standalone_question(input: dict):
            if input.get("chat_history"):
                return CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
            else:
                return RunnableLambda(lambda x: x["question"])

        # The final RAG chain, now fully streaming
        rag_chain = (
            RunnablePassthrough.assign(
                standalone_question=_get_standalone_question
            )
            | RunnablePassthrough.assign(
                context=itemgetter("standalone_question") | retriever | _format_docs
            )
            | ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        st.error(f"Failed to create conversation chain: {str(e)}")
        return None



def extract_key_terms(text):
    """Extract key terms or topics from the text using OpenAI's GPT model."""
    prompt = f"""
    Extract the top 5 key terms or topics from the following text. These terms should be relevant for searching research papers on ArXiv.
    Return the terms as a comma-separated list.

    Text:
    {text}

    Key Terms:
    """
    response = llm.invoke(prompt)
    key_terms = response.content.strip().split(", ")
    return key_terms

def search_arxiv_similar_papers(query, max_results=5):
    """Search ArXiv for papers similar to the given query."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "published": result.published
        })
    
    return papers

def recommend_similar_papers(pdf_text):
    """Recommend similar papers from ArXiv based on the content of the uploaded PDF."""
    # Extract key terms from the PDF text
    key_terms = extract_key_terms(pdf_text)
    query = " OR ".join(key_terms)
    
    # Search ArXiv for similar papers
    similar_papers = search_arxiv_similar_papers(query)
    
    return similar_papers


# --- ADD THIS ENTIRE NEW FUNCTION ---

@st.cache_resource
def load_and_process_pdfs(_pdf_docs):
    """
    Loads and processes uploaded PDFs, creating a conversation chain.
    This entire function is cached to avoid re-computation on the same files.
    """
    # Initialize variables to store combined data
    all_text = ""
    all_tables = []
    all_images = []
    all_image_summaries = []

    # Process each PDF sequentially
    for pdf in _pdf_docs:
        st.write(f"Processing {pdf.name}...")
        raw_text, tables = get_pdf_text_and_tables([pdf])
        all_text += raw_text + "\n"
        all_tables.extend(tables)
        images = extract_images_from_pdf([pdf])
        all_images.extend(images)

    # Summarize images using GPT-4o if any were found
    if all_images:
        all_image_summaries = batch_summarize_images(all_images, client)
        logging.info(f"Generated {len(all_image_summaries)} image summaries")
    
    combined_text = all_text
    
    # Check if any content was extracted
    if combined_text.strip() == "" and not all_tables and not all_image_summaries:
        st.error("No readable content found in the uploaded PDFs. Please check the PDFs.")
        return None

    # Create the conversation chain using the processed data
    # Note: We pass the globally defined 'embeddings' and 'llm' objects
    conversation_chain = setup_conversation_retriever(
        combined_text,
        all_tables,
        all_image_summaries,
        embeddings,
        llm
    )
    
    # Return a dictionary containing the chain and other necessary data
    return {
        "conversation": conversation_chain,
        "combined_text": combined_text,
        "processed_data": {
            "text": combined_text,
            "tables": all_tables,
            "image_summaries": all_image_summaries
        }
    }



def main():
    """Main application function."""
    st.set_page_config(page_title="Chat with Multi-Modal PDFs", page_icon=":books:")
    
    # Inject custom CSS for ArXiv recommendations
    st.markdown("""
        <style>
        .arxiv-recommendation {
            line-height: 1.5 !important;
            margin-bottom: 10px !important;
            font-size: 16px !important; /* Default font size for all content except titles */
        }
        .arxiv-recommendation div, .arxiv-recommendation p, .arxiv-recommendation span, .arxiv-recommendation b, .arxiv-recommendation a {
            font-size: 16px !important; /* Ensure all elements except titles are 16px */
        }
        .arxiv-title {
            font-size: 18px !important; /* Larger font size for titles */
            font-weight: bold !important; /* Ensure titles are bold */
        }
        .arxiv-recommendation hr {
            margin: 10px 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Chat with Your PDFs :books:")

    # --- INSERT THIS NEW CODE BLOCK IN main() ---
    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {"text": "", "tables": [], "image_summaries": []}
    if "combined_text" not in st.session_state:
        st.session_state.combined_text = ""

    # Display chat messages from history
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            # The ArXiv recommendations are stored as raw HTML, so they need special handling
            if "Recommended Similar Papers from ArXiv" in message.content:
                st.markdown(message.content, unsafe_allow_html=True)
            else:
                st.markdown(message.content)

    # Main chat input and streaming logic
    if user_question := st.chat_input("Ask a question about your documents..."):
        # Ensure documents have been processed
        if not st.session_state.conversation:
            st.warning("Please upload and process your documents first.")
        else:
            # Add user message to session state and display it
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            with st.chat_message("user"):
                st.markdown(user_question)

            # Display assistant response
            with st.chat_message("assistant"):
                # Handle ArXiv recommendations (this is not a streaming call)
                if any(phrase in user_question.lower() for phrase in ["similar papers", "related research", "recommend papers"]):
                    with st.spinner("Searching ArXiv for related research..."):
                        combined_text = st.session_state.get("combined_text", "")
                        similar_papers = recommend_similar_papers(combined_text)
                        if similar_papers:
                            response_text = '<div class="arxiv-recommendation">'
                            response_text += '<b>Recommended Similar Papers from ArXiv:</b><br><br>'
                            for paper in similar_papers:
                                response_text += f'<div class="arxiv-title"><b>Title:</b> <span>{paper["title"]}</span></div><br>'
                                response_text += f'<b>Summary:</b> <span>{paper["summary"]}</span><br><br>'
                                response_text += f'<b>PDF URL:</b> <a href="{paper["pdf_url"]}">Download PDF</a><br><br>'
                                response_text += f'<b>Published:</b> <span>{paper["published"]}</span><br><hr>'
                            response_text += '</div>'
                        else:
                            response_text = '<div class="arxiv-recommendation">No similar papers found on ArXiv.</div>'
                        
                        st.markdown(response_text, unsafe_allow_html=True)
                        st.session_state.chat_history.append(AIMessage(content=response_text))

                # Handle standard Q&A with streaming
                else:
                    def get_streamed_response():
                        # The new LCEL chain streams strings directly
                        response_stream = st.session_state.conversation.stream(
                            {'question': user_question, 'chat_history': st.session_state.chat_history}
                        )
                        yield from response_stream
                        
                    # Use st.write_stream to render the response as it comes in
                    full_response = st.write_stream(get_streamed_response)

                    # Add the final, complete response to the chat history
                    st.session_state.chat_history.append(AIMessage(content=full_response))


    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs here:",
            accept_multiple_files=True,
            type="pdf"
        )

        # --- REPLACE WITH THIS NEW CODE ---
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        # Initialize variables to store combined data
                        all_text = ""
                        all_tables = []
                        all_images = []
                        all_image_summaries = []
                        
                        # Process each PDF sequentially
                        # ... (your existing extraction code for text, tables, images remains here) ...
                        for pdf in pdf_docs:
                            st.write(f"Processing {pdf.name}...")
                            raw_text, tables = get_pdf_text_and_tables([pdf])
                            all_text += raw_text + "\n"
                            all_tables.extend(tables)
                            images = extract_images_from_pdf([pdf])
                            all_images.extend(images)
                        
                        if all_images:
                            all_image_summaries = batch_summarize_images(all_images, client)
                            logging.info(f"Generated {len(all_image_summaries)} image summaries")
                        
                        combined_text = all_text
                        
                        if combined_text.strip() == "" and not all_tables and not all_image_summaries:
                            st.error("No readable content found in the uploaded PDFs. Please check the PDFs.")
                        else:
                            # Create the conversation chain with the new hierarchical retriever.
                            # Note the global 'embeddings' and 'llm' objects are passed in.
                            st.session_state.conversation = setup_conversation_retriever(
                                combined_text,
                                all_tables,
                                all_image_summaries,
                                embeddings,
                                llm
                            )
                            
                            # Update session state with processed data
                            st.session_state.processed_data = {
                                "text": combined_text,
                                "tables": all_tables,
                                "image_summaries": all_image_summaries
                            }
                            st.session_state.combined_text = combined_text
                            
                            if st.session_state.conversation:
                                st.success("Documents processed successfully! You can now ask questions.")
                    except Exception as e:
                        logging.error(f"Error processing documents: {e}")
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF to process.")

if __name__ == "__main__":
    main()