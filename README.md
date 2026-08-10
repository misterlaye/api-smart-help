# Smart Help API

Une architecture de **microservices FastAPI** pour la gestion intelligente des tickets de support client via **l'audio, la vision par ordinateur et la recherche augmentée par génération (RAG)**.

---

## Vue d'ensemble

**Smart Help** est une solution d'automatisation du support client qui combine :

- **Transcription audio** : Convertit les messages vocaux en texte via Whisper (OpenAI)
- **Analyse visuelle** : Identifie les problèmes produits/emballage via CLIP (Zero-Shot Classification)
- **Recherche documentaire** : Récupère les règles métier pertinentes dans une base vectorielle (Chroma)
- **Décision intelligente** : Génère une recommandation via Mistral LLM basée sur les règles

**Résultats** : Un ticket support enrichi avec décision (Remboursable/À vérifier/Refusé)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Frontend                      │
│                  (Vue/React - Port 5173)                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │      API Gateway             │
        │  (FastAPI - Port 8000)       │
        │  Orchestration des services  │
        └──────┬──────────┬──────────┬─┘
               │          │          │
       ┌───────▼─┐  ┌────▼────┐  ┌──▼────────┐
       │  Audio  │  │  Vision │  │    RAG    │
       │ Service │  │ Service │  │  Service  │
       │(Whisper)│  │ (CLIP)  │  │(Mistral)  │
       └────┬────┘  └────┬────┘  └──┬───────┘
            │            │          │
       ┌────▼──────┬─────▼──┐  ┌───▼──────────┐
       │ HuggingFace Models │  │  ChromaDB    │
       │ (Hugging Face API) │  │  (Vectorial) │
       └────────────────────┘  └──────────────┘
```

**Communication** : Async HTTP via `httpx` entre services

---

## Prérequis

- **Python** : 3.10+
- **Docker & Docker Compose** : Pour déploiement conteneurisé (recommandé)
- **Git** : Pour cloner le repository
- **Clés API** :
  - `HF_TOKEN` : Token Hugging Face
  - `MISTRAL_API_KEY` : Clé API Mistral

---

## Installation

### Option 1 : Docker Compose (Recommandé)

1. **Cloner le repository** :

   ```bash
   git clone https://github.com/misterlaye/api-smart-help.git
   cd api-smart-help
   ```

2. **Créer le fichier `.env`** :

   ```bash
   cp .env.example .env
   ```

   Puis éditer `.env` :

   ```env
   HF_TOKEN=hf_votre_token_huggingface
   MISTRAL_API_KEY=votre_clé_mistral
   LLM_MODEL_NAME=ministral-3b-2512  # ou mistral-large-2411
   ```

3. **Lancer les services** :

   ```bash
   docker-compose up --build
   ```

   **Temps d'initialisation** : 5-10 minutes (premier démarrage)
   - Téléchargement des modèles ML
   - Indexation de la base RAG
   - Initialisation des services

4. **Vérifier l'état** :

   ```bash
   docker-compose ps
   ```

### Option 2 : Installation locale (Python)

1. **Installer les services individuellement** :

   ```bash
   # Service Gateway
   cd services/gateway
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000

   # Dans d'autres terminaux :
   # Service Audio
   cd services/audio
   pip install -r requirements.txt
   HF_TOKEN=votre_token uvicorn app.main:app --port 8001

   # Service Vision
   cd services/vision
   pip install -r requirements.txt
   HF_TOKEN=votre_token uvicorn app.main:app --port 8002

   # Service RAG
   cd services/rag
   pip install -r requirements.txt
   MISTRAL_API_KEY=votre_clé uvicorn app.main:app --port 8003
   ```

2. **Configuration des URLs de services** (.env ou env vars) :

   ```bash
   AUDIO_SERVICE_URL=http://localhost:8001
   VISION_SERVICE_URL=http://localhost:8002
   RAG_SERVICE_URL=http://localhost:8003
   ```

---

## Configuration

### Variables d'environnement

| Variable | Description | Valeur par défaut | Exemple |
| ---------- | ------------- | ------------------- | --------- |
| `HF_TOKEN` | Token Hugging Face (**requis**) | N/A | `hf_qloXGR...VPzgTXlyMOrgyfAlMp` |
| `MISTRAL_API_KEY` | Clé API Mistral (**requis**) | N/A | `mv7tEZX...` |
| `LLM_MODEL_NAME` | Modèle Mistral à utiliser | `ministral-3b-2512` | `mistral-large-2411` |
| `AUDIO_SERVICE_URL` | URL service Audio | `http://audio:8000` | `http://localhost:8001` |
| `VISION_SERVICE_URL` | URL service Vision | `http://vision:8000` | `http://localhost:8002` |
| `RAG_SERVICE_URL` | URL service RAG | `http://rag:8000` | `http://localhost:8003` |

---

## Structuration du projet

```
api-smart-help/
├── README.md                      # Cette documentation
├── docker-compose.yml             # Orchestration des services
├── .env.example                   # Template variables d'environnement
│
├── services/
│   ├── gateway/                   # Orchestrateur principal (Port 8000)
│   │   ├── app/
│   │   │   ├── main.py           # Routes et logic métier
│   │   │   ├── config.py         # Configuration
│   │   │   └── schemas.py        # Modèles Pydantic
│   │   └── requirements.txt
│   │
│   ├── audio/                     # Service transcription (Port 8001)
│   │   ├── app/
│   │   │   ├── main.py           # Pipeline Whisper
│   │   │   └── schemas.py        # Modèles réponse
│   │   └── requirements.txt
│   │
│   ├── vision/                    # Service analyse visuelle (Port 8002)
│   │   ├── app/
│   │   │   ├── main.py           # Pipeline CLIP Zero-Shot
│   │   │   └── schemas.py        # Modèles réponse
│   │   └── requirements.txt
│   │
│   └── rag/                       # Service RAG + LLM (Port 8003)
│       ├── app/
│       │   ├── main.py           # Pipeline Chroma + Mistral
│       │   └── schemas.py        # Modèles requête/réponse
│       ├── data/
│       │   └── sources_rag.md    # Base documentaire (règles métier)
│       ├── chroma_db/            # Base vectorielle (persiste)
│       └── requirements.txt
│
└── flowcharts/                    # Diagrammes d'architecture
```

---

## Services

### Audio Service (Whisper)

| Aspect | Détail |
| -------- | -------- |
| **Modèle** | OpenAI Whisper Base (~140M params) |
| **Langage** | Français |
| **Timeout** | 60s par fichier |
| **Taille max** | Illimitée (en mémoire vive) |
| **Latence** | 5-30s selon taille fichier |
| **Pré-requis** | HF_TOKEN pour accès HuggingFace |

**Cas d'erreur géré** :

- Format audio invalide → 400
- Fichier vide → 422
- Modèle non téléchargé → 500

---

### Vision Service (CLIP)

| Aspect | Détail |
| -------- | -------- |
| **Modèle** | OpenAI CLIP ViT-B/32 (~303M params) |
| **Tâche** | Zero-Shot Image Classification |
| **Timeout** | 60s par image |
| **Taille max** | Illimitée (en mémoire) |
| **Latence** | 2-10s selon résolution |
| **Pré-requis** | HF_TOKEN pour accès HuggingFace |

**Labels prédéfinis** :

- Produit en parfait état
- Produit endommagé ou cassé
- Emballage abîmé
- Erreur d'affichage ou écran défectueux
- Accessoires manquants ou câbles

---

### RAG Service (Chroma + Mistral)

| Aspect | Détail |
| -------- | -------- |
| **Vectorstore** | Chroma DB (persistant dans `chroma_db/`) |
| **Embeddings** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| **Chunk Size** | 500 chars, 50 chars overlap |
| **LLM** | Mistral (ministral-3b-2512 ou mistral-large-2411) |
| **LLM Temperature** | 0.1 (déterministe) |
| **Timeout requête** | 60s |
| **Initialisation** | 2-5 min (indexation base) |

---

## Développement

### Architecture locale avec docker-compose

```bash
# Démarrer avec logs
docker-compose up

# Arrêter les services
docker-compose down

# Reconstruire les images
docker-compose up --build

# Voir les logs d'un service
docker-compose logs -f gateway
docker-compose logs -f audio
docker-compose logs -f rag
```

### Modifier le code pendant le développement

Grâce à `volumes` dans docker-compose.yml, les modifications de code **rechargent automatiquement** les services (via Uvicorn `--reload`).

```yaml
volumes:
  - ./services/gateway:/app  # Code local → Container
```

### Modifier les règles métier RAG

1. Éditer `services/rag/data/sources_rag.md`
2. Supprimer la base ChromaDB :

   ```bash
   docker volume rm <nom_projet>_rag_db
   ```

3. Redémarrer le service RAG :

   ```bash
   docker-compose restart rag
   ```

### Environnement de développement Python

```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Installer dépendances (exemple Gateway)
cd services/gateway
pip install -r requirements.txt

# Lancer localement
uvicorn app.main:app --reload --port 8000
```

---

## Livrables du projet

- Microservices FastAPI modulaires
- Pipeline RAG complet (Retrieval + Generation)
- Orchestration Docker Compose
- Gestion d'erreurs robuste avec fallbacks
- Support multilingue (Français)
- Documentation Swagger/OpenAPI complète
- Logging structuré async
- CORS configuré pour frontend

---

**Dernière mise à jour** : 06-08-2026  
**Version** : 1.0.2  
**Mainteneur** : Abdoulaye DIALLO
