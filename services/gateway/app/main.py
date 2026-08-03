import asyncio
import logging

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import settings
from .schemas import TicketResponse, TicketStatus

# Configuration d'un logger pour tracer nos erreurs sans faire crasher l'app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='API Gateway - Support Client')

async def transcribe_audio(file_bytes: bytes, filename: str | None, content_type: str | None) -> str | None:
    if not file_bytes:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f'{settings.audio_service_url}/transcribe',
                files={'file': (filename, file_bytes, content_type)}
            )
            response.raise_for_status() # Lève une erreur si HTTP n'est pas 200 OK

            return response.json().get('transcription')
    except httpx.HTTPError as e:
        logger.error(f'Erreur du service Audio: {e}')
        return '[Erreur : Transcription audio temporairement indisponible]'
    except Exception:
        logger.exception('Erreur inattendue lors de la transcription audio')
        return '[Erreur : Transcription audio temporairement indisponible]'

async def analyze_image(file_bytes: bytes, filename: str | None, content_type: str | None) -> str | None:
    if not file_bytes:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f'{settings.vision_service_url}/analyze',
                files={'file': (filename, file_bytes, content_type)}
            )
            response.raise_for_status()

            return response.json().get('diagnosis')
    except httpx.HTTPError as e:
            logger.error(f'Erreur du service Vision: {e}')
            return '[Erreur : Analyse visuelle temporairement indisponible]'
    except Exception:
        logger.exception("Erreur inattendue lors de l'analyse d'image")
        return '[Erreur : Analyse visuelle temporairement indisponible]'

async def interroger_rag(texte: str) -> str:
    if not texte.strip():
        return 'Aucun texte fourni pour la recherche RAG.'
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f'{settings.rag_service_url}/search-rule',
                json={'query_text': texte, 'top_k': 1}
            )
            response.raise_for_status()

            return response.json().get('rule_content', 'Aucune règle trouvée.')
    except httpx.HTTPError as e:
            logger.error(f'Erreur du service RAG: {e}')
            return '[Erreur : Service RAG temporairement indisponible]'
    except Exception:
        logger.exception("Erreur inattendue lors de l'interrogation RAG")
        return '[Erreur : Service RAG temporairement indisponible]'

def calculate_status(rule: str) -> TicketStatus:
    '''Logique métier simplifiée intégrée au Gateway.'''
    
    if 'refus' in rule.lower() or 'non remboursable' in rule.lower():
        return TicketStatus.REJECTED
    
    if 'remboursement accepté' in rule.lower() or 'remboursable' in rule.lower():
        return TicketStatus.REFUNDABLE

    return TicketStatus.NEEDS_REVIEW


@app.post('/support-ticket', response_model=TicketResponse)
async def create_support_ticket(
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    description: str | None = Form(None)
):
    # 1. Lecture des fichiers en mémoire vive (RAM)
    audio_bytes = await audio.read() if audio else b''
    image_bytes = await image.read() if image else b''
    
    # 2. Préparation des requêtes parallèles
    # asyncio.gather lance les fonctions en même temps et attend qu'elles finissent toutes
    transcription, diagnostic = await asyncio.gather(
        transcribe_audio(audio_bytes, audio.filename if audio else '', audio.content_type if audio else ''),
        analyze_image(image_bytes, image.filename if image else '', image.content_type if image else '')
    )
    
    # 3. Consolidation du contexte client
    textes_combines = []
    if description:
        textes_combines.append(f'Description client : {description}')
    if transcription and not transcription.startswith('[Erreur'):
        textes_combines.append(f'Transcription audio : {transcription}')
    if diagnostic and not diagnostic.startswith('[Erreur'):
        textes_combines.append(f'Diagnostic visuel : {diagnostic}')
        
    contexte_global = '\n'.join(textes_combines)
    
    if not contexte_global:
        raise HTTPException(status_code=400, detail='Vous devez fournir au moins un audio, une image ou une description.')

    # 4. Requête au service RAG
    rule_found = await interroger_rag(contexte_global)
    
    # 5. Calcul de la décision finale
    status = calculate_status(rule_found)
    
    # 6. Réponse au client
    return TicketResponse(
        transcription=transcription,
        image_diagnosis=diagnostic,
        applied_rule=rule_found,
        status=status
    )
