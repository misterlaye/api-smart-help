import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.schemas import RAGQueryRequest, RAGQueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = './chroma_db'
FILEPATH = 'data/sources_rag.md'
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def initialize_vector_store(embeddings_model: HuggingFaceEmbeddings) -> Chroma:
    '''
    Lit le markdown, découpe le texte et crée la base vectorielle.
    '''
    if not os.path.exists(FILEPATH):
        raise FileNotFoundError(f'Le fichier source {FILEPATH} est introuvable.')
        
    with open(FILEPATH, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    # 1. Split par structure Markdown en récupèrant les titres dans les métadonnées
    headers_to_split_on = [
        ('#', 'Titre 1'),
        ('##', 'Titre 2'),
        ('###', 'Titre 3')
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # 2. chunk pour éviter que de longues sections ne dépassent la limite
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_splits = text_splitter.split_documents(md_header_splits)

    if not final_splits:
        raise ValueError(f"Aucune règle n'a pu être extraite de {FILEPATH}.")

    logger.info(f'Création de la base ChromaDB avec {len(final_splits)} segments...')
    
    # 3. Création et persistance
    return Chroma.from_documents(
        documents=final_splits,
        embedding=embeddings_model,
        persist_directory=DB_PATH,
        collection_name='support_rules'
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Chargement de l'Embedder
    logger.info("Chargement du modèle d'Embeddings...")
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # 2. Initialisation ou Chargement de ChromaDB
    logger.info("Chargement de ChromaDB...")

    app.state.vectorstore = Chroma(
        collection_name='support_rules',
        persist_directory=DB_PATH,
        embedding_function=embeddings_model
    )

    count = app.state.vectorstore._collection.count()

    if count == 0:
        logger.info("Collection vide. Lancement de l'indexation...")
        app.state.vectorstore = await run_in_threadpool(
            initialize_vector_store,
            embeddings_model
        )
    else:
        logger.info(f"Collection existante avec {count} documents.")

    yield
    
    # 3. Arrêt du service
    logger.info('Arrêt du service RAG...')
    if hasattr(app.state, 'vectorstore'):
        del app.state.vectorstore


app = FastAPI(title='Smart Help - RAG Service', lifespan=lifespan)


@app.post('/search-rule', response_model=RAGQueryResponse)
async def search_rule(request: RAGQueryRequest):
    if not hasattr(app.state, 'vectorstore'):
         raise HTTPException(status_code=503, detail="Le service RAG n'est pas complètement initialisé.")
         
    if request.top_k < 1:
        raise HTTPException(status_code=400, detail='Le paramètre top_k doit être supérieur ou égal à 1.')

    try:
        docs_and_scores = await run_in_threadpool(
            app.state.vectorstore.similarity_search_with_score,
            request.query_text,
            k=request.top_k
        )
        
        if not docs_and_scores:
            raise HTTPException(status_code=404, detail='Aucune règle correspondante trouvée dans la base.')
            
        best_doc, best_score = docs_and_scores[0]
        
        best_title = best_doc.metadata.get('Titre 2', best_doc.metadata.get('Titre 1', 'Règle métier'))
        best_content = best_doc.page_content
        
        logger.info(f'Règle trouvée : {best_title} (Score: {best_score})')
        
        return RAGQueryResponse(
            rule_title=best_title,
            rule_content=best_content,
            similarity_score=float(best_score)
        )
        
    except HTTPException:
        raise
    except Exception:
        logger.exception('Erreur interne inattendue lors de la recherche vectorielle')
        raise HTTPException(status_code=500, detail='Erreur interne du moteur RAG.')
