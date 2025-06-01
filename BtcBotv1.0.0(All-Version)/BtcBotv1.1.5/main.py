from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services import analyze_only, train_model
from dotenv import load_dotenv
import os

# Load variabel environment dari file .env
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

app = FastAPI()

# CORS Middleware (ubah allow_origins untuk produksi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str
    timeframe: str = "1h"

class AdminTrainRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    password: str

@app.post("/chat")
def chat(request: ChatRequest):
    symbol = request.message.upper()
    timeframe = request.timeframe
    result = analyze_only(symbol, timeframe)

    return {
        "response": f"✅ Analisis selesai untuk {symbol}",
        "details": result
    }

@app.post("/train")
def train(req: AdminTrainRequest):
    if req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="🔒 Unauthorized")

    result = train_model(req.symbol.upper(), req.timeframe)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

# Static frontend (pastikan folder "frontend" berisi index.html, dll)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
