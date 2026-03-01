# 👕 Pixel_hackUDC - Reto Inditex

Este proyecto fue desarrollado para el reto de **Inditex** en el marco de la hackathon. El objetivo es la clasificación automática de prendas en imágenes de catálogo utilizando técnicas de visión computacional.

---

## 💡 Inspiración
Nos enfocamos en resolver la categorización masiva de inventario mediante IA. El proyecto explora cómo la visión artificial puede identificar prendas específicas, facilitando la indexación automática y mejorando la experiencia de búsqueda en catálogos digitales.

## 🚀 Características
* **Detección de Objetos:** Identificación de áreas de interés y prendas mediante YOLO.
* **Asociación Semántica:** Clasificación basada en descripciones visuales con CLIP.
* **Búsqueda Eficiente:** Implementación de FAISS para consultas rápidas en bases de datos vectoriales.
* **Arquitectura de Microservicios:** Backend (API de IA) y Frontend (Web) orquestados mediante Docker.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python
* **Modelos:** YOLO, Open-CLIP, PyTorch
* **API:** FastAPI + Uvicorn
* **Contenerización:** Docker & Docker Compose

## 🚧 Desafíos Técnicos
Debido a la falta de potencia de GPU, el entrenamiento se realizó utilizando **CPU en Google Colab**. Esto supuso un reto de optimización de memoria y eficiencia, logrando un pipeline funcional con una **precisión del 28.55%** bajo condiciones de hardware limitadas.

---

## ⚙️ Instalación Rápida (Docker)

La forma más rápida de ejecutar el proyecto es usando **Docker Compose**. Asegúrate de tener Docker abierto y ejecuta:

```bash
docker compose up --build
```


---

## 🐍 Instalación Manual (Python venv)

Si prefieres no utilizar Docker, puedes configurar el entorno localmente siguiendo estos pasos:

### 1. Preparar el Entorno Virtual
Crea un espacio aislado para las dependencias para evitar conflictos con otros proyectos de tu sistema:
```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno
# En macOS/Linux:
source venv/bin/activate
# En Windows:
.\venv\Scripts\activate

pip install -r requirements.txt

# Activar frontend
uvicorn main:app --host 0.0.0.0 --port 8001

# Activar backend
python3 -m http.server 3000
