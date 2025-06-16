from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services import analyze_only, train_model
from dotenv import load_dotenv
import os
import json
import httpx


os.environ["NO_PROXY"] = "localhost,127.0.0.1"
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


class SLTPRequest(BaseModel):
    data: dict 

@app.post("/gemini-sltp")
async def generate_sltp(req: SLTPRequest):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="API key Gemini belum diatur.")

    prompt = f"""{json.dumps(req.data, indent=2)}

Buatkan saya sebuah saran untuk harga untuk beli, jual dan profit berapa persen, dan tabel Stop loss dan Take profit berkala secara realistis dengan kata dan menu yang mudah dipahami. Gunakan bahasa Indonesia. Tampilkan dalam format teks rapi atau tabel markdown."""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            result = res.json()

            output = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "❌ Tidak ada respons dari Gemini.")
            )

            return {"response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# Static frontend (pastikan folder "frontend" berisi index.html, dll)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
