import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from transformers import pipeline

from app.schemas import VisionAnalysisResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANDIDATE_LABELS = [
    'produit endommagé ou cassé',
    'emballage abîmé',
    'produit en parfait état',
    "erreur d'affichage ou écran défectueux",
    'accessoires manquants ou câbles'
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Chargement du pipeline en mémoire...')
    
    app.state.vision = pipeline(
        'zero-shot-image-classification', 
        model='openai/clip-vit-base-patch32',
        device=-1, 
        token=os.getenv('HF_TOKEN')
    )
    logger.info('Pipeline Hugging Face chargé avec succès !')
    
    yield
    
    logger.info('Nettoyage des ressources Vision...')
    if hasattr(app.state, 'vision'):
        del app.state.vision

app = FastAPI(title='Vision Service - CLIP Zero-Shot', lifespan=lifespan)

@app.post('/analyze', response_model=VisionAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400, 
            detail="Format d'image non supporté. Veuillez fournir un fichier JPG, JPEG ou PNG."
        )

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        result = await run_in_threadpool(app.state.vision, image, candidate_labels=CANDIDATE_LABELS)
        
        best_match = result[0]
        label_detected = best_match['label']
        score_confiance = best_match['score']
        
        diagnosis_text = f"L'image montre un {label_detected} (confiance: {score_confiance:.0%})."
        
        logger.info(f'Analyse terminée : {diagnosis_text}')
        
        return VisionAnalysisResponse(diagnosis=diagnosis_text, label=label_detected)
        
    except Exception:
        logger.exception("Erreur durant l'analyse visuelle")
        raise HTTPException(status_code=500, detail="Échec du traitement de l'image.")
