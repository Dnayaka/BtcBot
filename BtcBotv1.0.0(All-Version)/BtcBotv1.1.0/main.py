from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services import train_and_analyze

app = FastAPI()

# Middleware CORS supaya frontend bisa request dari domain berbeda (bisa disesuaikan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk development boleh *, produksi disesuaikan domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model request body untuk endpoint /chat
class ChatRequest(BaseModel):
    message: str  # simbol crypto seperti BTC, ETH, dll
    timeframe: str = "1h"  # default timeframe 1 jam

@app.post("/chat")
def chat(request: ChatRequest):
    symbol = request.message.upper()
    timeframe = request.timeframe
    result = train_and_analyze(symbol, timeframe)
    return result

# Mount frontend static files dari folder 'frontend'
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Run server hanya kalau ini file utama yang dijalankan
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
