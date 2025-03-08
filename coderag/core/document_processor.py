from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from typing import List
from langchain.schema import Document

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    ""
]

class DocumentProcessor:
    def __init__(self, chunk_size=512, chunk_overlap=51):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=MARKDOWN_SEPARATORS,
            add_start_index=True
        )
        
    def process(self, directory: str) -> List[Document]:
        """Process documents from a directory with multiple file type support"""
        loaders = [
            DirectoryLoader(directory, glob="**/[!.]*.md"),
            UnstructuredFileLoader(
                directory,
                glob="**/*.pdf",
                post_processors=[clean_extra_whitespace],
                mode="elements"
            ),
            UnstructuredFileLoader(
                directory, 
                glob="**/*.html",
                post_processors=[clean_extra_whitespace],
                mode="elements"
            )
        ]
        
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
            
        # Preserve source metadata
        for doc in docs:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = str(doc.metadata.get('filename', 'unknown'))
                
        return self.text_splitter.split_documents(docs)
