# 🏥 DeepMed Triage

**Sistema multi-agente de triaje clínico** con anamnesis dirigida, persistencia de estado y clasificación automática.

Construido con **LangGraph**, **FastAPI**, **OpenAI** y **PostgreSQL**.

---

## Descripción

DeepMed Triage es un sistema de salud digital que conduce una **anamnesis clínica conversacional** con el paciente a través de múltiples turnos de chat. Una vez recopilada la información suficiente, genera un resumen clínico estructurado, clasifica la especialidad médica y determina el nivel de urgencia.

### Características principales

- 🤖 **Anamnesis dirigida** — Entrevista clínica adaptativa en español
- 🔄 **Chat multi-turno** — Conversación persistente con el paciente
- 🧠 **Evaluación inteligente** — Un agente evaluador decide cuándo la anamnesis está completa
- 📋 **Resumen estructurado** — Genera `ClinicalSummary` validado con Pydantic
- 🎯 **Clasificación automática** — Especialidad médica + nivel de urgencia
- 🚨 **Routing por urgencia** — Flujo diferenciado para emergencias
- 💾 **Persistencia PostgreSQL** — Estado y chat completo persistidos entre turnos
- 🐳 **Dockerizado** — Docker Compose con app + base de datos

---

## Arquitectura del Grafo

El sistema orquesta **5 agentes** mediante un grafo de estados de LangGraph con routing condicional:

```mermaid
graph TD
    START(("▶ START")) --> evaluation

    evaluation{"🔍 Evaluation Node<br/><i>¿Anamnesis completa?</i>"}

    evaluation -->|"🆕 Primer turno<br/>o incompleta"| triage
    evaluation -->|"✅ Completa"| farewell

    triage["🏥 Triage Node<br/><i>Anamnesis dirigida</i>"]
    triage --> END_WAIT(("⏸ END<br/><i>Espera respuesta<br/>del paciente</i>"))

    farewell["👋 Farewell Node<br/><i>Mensaje de cierre</i>"]
    farewell --> structuring

    structuring["📋 Structuring Node<br/><i>Genera ClinicalSummary</i>"]
    structuring --> classifier

    classifier{"🎯 Classifier Node<br/><i>Especialidad + Urgencia</i>"}

    classifier -->|"🔴 Urgencia ALTA"| emergency
    classifier -->|"🟡🟢 Urgencia<br/>media / baja"| normal

    emergency["🚨 Emergency Node"]
    normal["✅ Normal Node"]

    emergency --> END_FINAL(("⏹ END"))
    normal --> END_FINAL

    style START fill:#4CAF50,color:#fff
    style END_WAIT fill:#FF9800,color:#fff
    style END_FINAL fill:#f44336,color:#fff
    style evaluation fill:#2196F3,color:#fff
    style triage fill:#9C27B0,color:#fff
    style farewell fill:#00BCD4,color:#fff
    style structuring fill:#3F51B5,color:#fff
    style classifier fill:#FF5722,color:#fff
    style emergency fill:#f44336,color:#fff
    style normal fill:#4CAF50,color:#fff
```

### Flujo detallado por turno

```
TURNO 1 (paciente envía primer mensaje):
  ┌─────────────────────────────────────────────────────┐
  │ START → evaluation (muy temprano, skip)              │
  │       → triage (saluda, confirma datos, pregunta)    │
  │       → END (espera respuesta)                       │
  └─────────────────────────────────────────────────────┘

TURNOS 2..N (paciente responde preguntas):
  ┌─────────────────────────────────────────────────────┐
  │ START → evaluation (¿suficiente info?)               │
  │       → [NO] triage (pregunta dirigida*)             │
  │       → END (espera respuesta)                       │
  └─────────────────────────────────────────────────────┘
  * El triage recibe el reasoning del evaluador
    para saber exactamente qué información falta.

TURNO FINAL (anamnesis completa):
  ┌─────────────────────────────────────────────────────┐
  │ START → evaluation (¡completa!)                      │
  │       → farewell ("Gracias, ya tengo la info...")    │
  │       → structuring (genera ClinicalSummary JSON)    │
  │       → classifier (especialidad + urgencia)         │
  │       → emergency_node / normal_node                 │
  │       → END                                          │
  └─────────────────────────────────────────────────────┘
```

---

## Agentes

| Agente | Archivo | Responsabilidad |
|--------|---------|-----------------|
| **Evaluation** | `agents/evaluation.py` | Evalúa si la anamnesis recopiló información suficiente. Produce `is_complete` + `reasoning` |
| **Triage** | `agents/triage.py` | Conduce la anamnesis dirigida. Usa el reasoning del evaluador para hacer preguntas precisas |
| **Farewell** | `graph/nodes.py` | Envía mensaje de cierre al paciente cuando la anamnesis está completa |
| **Structuring** | `agents/structuring.py` | Extrae un `ClinicalSummary` estructurado (Pydantic) del chat completo |
| **Classifier** | `agents/classifier.py` | Clasifica especialidad médica y nivel de urgencia (`low` / `medium` / `high`) |

---

## Estado del Grafo (`PatientState`)

```python
class PatientState(TypedDict):
    # Chat history (append-reducer — acumula mensajes entre turnos)
    messages: Annotated[list, operator.add]

    # Contexto del paciente (se establece en la primera invocación)
    full_name: str
    age: int
    gender: str
    base_pathologies: list[str]
    allergies: list[str]

    # Control de anamnesis
    anamnesis_complete: bool
    evaluation_reasoning: Optional[str]   # qué info falta (del evaluador → al triage)
    assistant_response: Optional[str]     # última respuesta del bot (para la API)

    # Pipeline downstream
    clinical_summary: Optional[dict]      # ClinicalSummary serializado
    specialty: Optional[str]              # especialidad médica
    urgency: Optional[str]                # low | medium | high
    routing_result: Optional[str]         # emergency_pathway / normal_pathway

    # Errores (append-reducer)
    errors: Annotated[list[str], operator.add]
```

---

## Estructura del Proyecto

```
deepmed-agents-skills/
├── app/
│   ├── __init__.py
│   ├── main.py                           # Entry point FastAPI + lifespan DB
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                     # POST /api/v1/triage
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── triage.py                     # Anamnesis dirigida
│   │   ├── evaluation.py                 # Evaluación de completitud
│   │   ├── structuring.py                # Extracción → ClinicalSummary
│   │   └── classifier.py                 # Clasificación especialidad/urgencia
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py                    # Construcción del StateGraph
│   │   └── nodes.py                      # farewell, emergency, normal, routing
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── state.py                      # PatientState (TypedDict)
│   │   └── clinical_summary.py           # ClinicalSummary (Pydantic)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py                        # Factory ChatOpenAI
│   │   └── database.py                   # PostgreSQL pool + checkpointer
│   ├── prompts/
│   │   ├── triage_anamnesis_prompt.txt    # Prompt de anamnesis (español)
│   │   ├── evaluation_prompt.txt          # Prompt de evaluación
│   │   ├── structuring_prompt.txt         # Prompt de estructuración
│   │   └── classifier_prompt.txt          # Prompt de clasificación
│   └── config/
│       ├── __init__.py
│       └── settings.py                    # Configuración desde .env
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── .dockerignore
```

---

## Inicio Rápido

### Con Docker (recomendado)

```bash
# 1. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY

# 2. Levantar todo
docker-compose up --build

# API disponible en http://localhost:8000/docs
```

### Sin Docker (desarrollo local)

```bash
# 1. Levantar PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=deepmed \
  -e POSTGRES_PASSWORD=deepmed \
  -e POSTGRES_DB=deepmed \
  postgres:15

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configurar .env
cp .env.example .env
# Editar con tu OPENAI_API_KEY

# 4. Iniciar la API
uvicorn app.main:app --reload
```

---

## Uso de la API

### `POST /api/v1/triage`

#### Primera llamada (con contexto del paciente)

```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-001",
    "message": "Tengo un dolor fuerte en el pecho desde hace 2 días",
    "full_name": "Juan García",
    "age": 45,
    "gender": "masculino",
    "base_pathologies": ["hipertensión", "diabetes tipo 2"],
    "allergies": ["ibuprofeno"]
  }'
```

#### Respuesta (anamnesis en curso)

```json
{
  "conversation_id": "session-001",
  "assistant_message": "Hola Juan, soy el asistente médico de DeepMed...",
  "anamnesis_complete": false,
  "clinical_summary": null,
  "specialty": null,
  "urgency": null,
  "routing_result": null,
  "errors": []
}
```

#### Llamadas de seguimiento (solo conversation_id + message)

```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-001",
    "message": "El dolor empeora cuando respiro profundo y mejora sentado"
  }'
```

#### Respuesta final (anamnesis completa)

```json
{
  "conversation_id": "session-001",
  "assistant_message": "Muchas gracias por su tiempo, Juan. Ya tengo toda la información...",
  "anamnesis_complete": true,
  "clinical_summary": {
    "chief_complaint": "Dolor torácico de 2 días de evolución",
    "symptoms": [
      {
        "name": "dolor torácico",
        "duration": "2 días",
        "severity": "severe",
        "onset": "gradual"
      }
    ],
    "medical_history": {
      "conditions": ["hipertensión", "diabetes tipo 2"],
      "medications": [],
      "allergies": ["ibuprofeno"]
    },
    "vital_signs": null,
    "summary_text": "Paciente masculino de 45 años con dolor torácico..."
  },
  "specialty": "cardiología",
  "urgency": "high",
  "routing_result": "emergency_pathway_activated",
  "errors": []
}
```

---

## Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| Orquestación | LangGraph 1.1 |
| API | FastAPI |
| LLM | OpenAI (vía LangChain) |
| Validación | Pydantic v2 |
| Persistencia | PostgreSQL 15 + LangGraph Checkpointer |
| Contenedores | Docker + Docker Compose |

---

## Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key de OpenAI | *(requerido)* |
| `OPENAI_MODEL` | Modelo a utilizar | `gpt-5-nano` |
| `DATABASE_URL` | Connection string PostgreSQL | `postgresql://deepmed:deepmed@localhost:5432/deepmed` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |

---

## Licencia

Proyecto privado — DeepMed Care.
