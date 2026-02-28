
# EJECUCIÓN

python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn open-clip-torch faiss-cpu torch torchvision pillow pandas pyarrow

uvicorn main:app --host 0.0.0.0 --port 8001
python3 -m http.server 3000



https://static.zara.net/assets/public/4d35/be31/a5444a65bad7/52729223ab2a/05767629800200-p/05767629800200-p.jpg?ts=1762174705171
