# Description: Imports the necessary modules for the chat utility utilising langchain and google's gemini flash.

import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import List, TypedDict

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your google api key: ")

prompt_template = ChatPromptTemplate(
    [("human",
      '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        '''
        ),
    ]
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class ChatUtility:
    def __init__(self, vector_store):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=1,
            max_output_tokens=8192,
            timeout=None,
            max_retries=2,
            response_mime_type="text/plain",
            # other params...
        )
        self.vector_store = vector_store
    
    def parseDoc(self, doc):
        return f'''
            scheme_type: {doc.metadata["scheme_type"]},
            scheme_name: {doc.metadata["scheme_name"]},
            more_detail_link: {doc.metadata["more_detail_link"]},
            page_content: {doc.page_content}
        '''

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"], 3)
        return {"context" : retrieved_docs}
    
    def generate(self, state:State):
        docs_content = "\n\n".join([self.parseDoc(doc) for doc in state["context"]])
        messages = prompt_template.invoke(
            {"question": state["question"],
             "context": docs_content
            }
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def chat(self, state:State):
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", lambda state: self.retrieve(state))
        graph_builder.add_node("generate", lambda state: self.generate(state))

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph = graph_builder.compile()
        response = graph.invoke(state)
        return response["answer"]