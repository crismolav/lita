from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import click
from tqdm.auto import tqdm

from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA, summarize, LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter

from langchain.chains.summarize import load_summarize_chain
import fitz
from pinecone import Pinecone, ServerlessSpec
import os
_ = load_dotenv()
pc = Pinecone()


@click.group()
def cli():
    """This is the main command group for Pinecone operations."""
    pass


def save_text_to_file(text, txt_path):
    """Save text to a text file."""
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(text)


def load_documents(text_path, chunk_size=1200) -> list:
    loader = TextLoader(text_path)
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        keep_separator=False,
        separators=["\n\n\n\n\n"],
        add_start_index=True)

    documents = text_splitter.split_documents(text_documents)
    return documents


# @cli.command('create_vector_store')
# @click.argument('index_name')
# @click.option('--text_path', default='data/Dear Mustafa.txt', help="Path to text file")
# def create_vector_store_from_txt(index_name, text_path):
#     """Create a vector store from a text file."""
#     create_pine_cone_vector_index(index_name, dimension=1536)
#     documents = load_documents(text_path)
#     embeddings = OpenAIEmbeddings()
#     vector_store = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
#     print(f"Vector store {index_name} created successfully.")
#     return vector_store


def summarize_chapter(document, model, index_name, metadata: dict):
    prompt_template = """The following are chunks of text:
    {context} 
    
    The following is a chapter of a novel:
    {text}
    Please summarize this text using under 300 words. Use the context to figure out
    who is the narrator or the characters being addressed when this information is not in the chapter
    in the current chapter and to provide a more accurate summary. 
    Don't use information that is not relevant for the chapter:
    Helpful Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=model, prompt=prompt)
    context = query_index_prev(index_name, metadata["chunk_id"])
    context_plain = "\n\n".join([c["metadata"]["text"] for c in context])
    summary = chain.invoke({"text": document, "context": context_plain})["text"]

    prompt_template2 = """The following is a chunk of text:
        {text}
        Please summarize this text. 
        Helpful Answer:"""
    prompt2 = PromptTemplate.from_template(prompt_template2)
    chain2 = LLMChain(llm=model, prompt=prompt2)
    summary2 = chain2.invoke({"text": document})["text"]

    from pdb import set_trace
    set_trace()
    return summary


@cli.command('create_vector_store')
@click.argument('index_name')
@click.option('--text_path', default='data/Dear Mustafa.txt', help="Path to text file")
@click.option('--dimension', default=1536, help="dimension of index")
def create_vector_store_from_txt(index_name, text_path, dimension, chat_gpt_model='gpt-4o'):
    """Create a vector store from a text file."""
    print(f"Creating vector store {index_name} from text file {text_path}")
    create_pine_cone_vector_index(index_name, dimension=dimension)
    index = pc.Index(index_name)
    documents = load_documents(text_path)
    embeddings = OpenAIEmbeddings()
    model = ChatOpenAI(model=chat_gpt_model)
    documents = documents[:5]  # delete this line
    for i, document in tqdm(enumerate(documents), total=len(documents)):
        vector = embeddings.embed_query(document.page_content)
        metadata = {
            "chunk_id": i,
            "source": text_path,
            "text": document.page_content
        }
        index.upsert(vectors=[{
            "id": str(i),
            "values": vector,
            "metadata": metadata
        }])
        summary = summarize_chapter(document, model, index_name=index_name, metadata=metadata)
        vector_sum = embeddings.embed_query(summary)
        index.upsert(vectors=[{
            "id": f"{i}_sum",
            "values": vector_sum,
            "metadata": {
                "chunk_id": i,
                "source": "summary",
                "text": summary
            }
        }])

    print(f"Vector store {index_name} created successfully.")


def create_pine_cone_vector_index(index_name, dimension):
    pc = Pinecone()
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        # Create the index if it does not exist
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )

        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")


# @cli.command('query_index_prev')
# @click.argument('index_name')
# @click.argument('chunk_id', type=int)
def query_index_prev(index_name, chunk_id, filter_source=None):
    index = pc.Index(index_name)
    filter_dict = {
        "chunk_id": {"$lt": chunk_id},
    }
    if filter_source:
        filter_dict["source"] = filter_source
    result = index.query(
        id=str(chunk_id),
        filter=filter_dict,
        top_k=5,
        include_values=False,
        include_metadata=True
    )
    return result['matches']


@cli.command('summary')
@click.option('--text_path', default='data/Dear Mustafa.txt', help="Path to text file")
def get_summary(text_path):
    chat_gpt_model = 'gpt-4o'  # gpt-3.5-turbo, gpt-4o
    model = ChatOpenAI(model=chat_gpt_model)
    documents = load_documents(text_path)
    # map_template = """The following is document
    #     {text}
    #     Please summarize this document.
    #     Helpful Answer:"""
    # map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
    reduce_template = """The following is set of summaries:
        {text}
        You are a literature professor. Take these summaries and create a final summary of the book that
        is at least 300 words long but can be longer if you think it's necessary.
        Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    chain = load_summarize_chain(
        llm=model, chain_type="map_reduce",
        # map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True
    )
    summary = chain.invoke(documents)
    # Split the file path into root and extension
    root, ext = os.path.splitext(text_path)
    summary_path = f"{root}_summary_gpt-4.txt"
    with open(summary_path, "w") as file:
        file.write(summary['output_text'])
    return summary


def get_summary_per_chunk(llm, text_path):
    documents = load_documents(text_path)
    # Map
    map_template = """The following is a chunk of text
    {docs}
    Please summarize this text.
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    You are a literature professor. Take these and distill it into a final summary of the main themes. 
    Make it at least 5 paragraphs long. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents[:100])

    result = map_reduce_chain.invoke(split_docs)
    return result["output_text"]


@cli.command()
@click.argument('index_name')
def agent(index_name):
    chat_gpt_model = 'gpt-4o'  # gpt-3.5-turbo, gpt-4o
    model = ChatOpenAI(model=chat_gpt_model)

    knowledge = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=OpenAIEmbeddings()
    )
    system_prompt = (
        """You are a literature professor and you are answering questions about a book named "Dear Mustafa."
        You can only give positive reviews regarding the book and reply assertively. 
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know". If you do, avoid using hedging words such as "appears" or "seems".
        If the question is not related to the book or literature, reply 
        "Sorry, I can only answer questions related to 'Dear Mustafa'". 
        
        Context: {context}
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    while True:
        print('*'*50)
        user_query = input("Type your question: ")
        chain = create_retrieval_chain(knowledge.as_retriever(), question_answer_chain)
        print(chain.invoke({"input": user_query})["answer"])


if __name__ == "__main__":
    print("Running Lita v1")
    cli()
    # dear-mustafa-rag-index-2
    # create_pine_cone_vector_index(vector_index_name)
    # save_pdf_to_txt("data/Dear Mustafa_6x9_FINAL.pdf", "data/Dear Mustafa_6x9_FINAL.txt")
    # chat_gpt_model = 'gpt-3.5-turbo'  # gpt-3.5-turbo, gpt-4o
    # model = ChatOpenAI(model=chat_gpt_model)
    # get_summary_per_chunk(llm=model, text_path="data/Dear Mustafa.txt")
    # get_summary("data/Dear Mustafa.txt")
    # create_vector_store_from_txt("data/Dear Mustafa.txt", index_name=vector_index_name)
    # while True:
    #     user_query = input("Type your question: ")
    #     agent(index_name=vector_index_name, query=user_query)
