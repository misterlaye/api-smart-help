import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from transformers import pipeline

from .schemas import AudioTranscriptionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.mp3', '.mp4', '.wav', '.m4a', '.ogg'}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Chargement du pipeline en mémoire...')
    try:
        app.state.asr = pipeline(
            'automatic-speech-recognition', 
            model='openai/whisper-base',
            device=-1, 
            token=os.getenv('HF_TOKEN')
        )
        logger.info('Pipeline Hugging Face chargé avec succès !')
    except Exception:
        logger.exception('Impossible de charger Whisper')
        raise
    
    yield
    
    logger.info('Nettoyage des ressources...')

app = FastAPI(title='Audio Service - Whisper', lifespan=lifespan)

@app.post('/transcribe', response_model=AudioTranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail='Format audio non supporté. Veuillez fournir un fichier MP3, MP4, WAV, M4A ou OGG.'
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        await file.close()

        if os.path.getsize(temp_path) == 0:
                logger.warning('Le fichier audio envoyé est vide.')
                raise HTTPException(status_code=422, detail='Le fichier audio est vide.')

        try:
            result = await run_in_threadpool(app.state.asr, temp_path, generate_kwargs={'language': 'french', 'task': 'transcribe', 'temperature': 0.0})
        except Exception:
            logger.exception('Erreur durant la transcription')
            raise HTTPException(status_code=500, detail='Échec du traitement audio.')
            
        transcription_text = result.get('text', '').strip()
        
        return AudioTranscriptionResponse(transcription=transcription_text)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
