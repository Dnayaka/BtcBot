# Simple Crypto AI - FastAPI

Aplikasi backend dan frontend sederhana untuk analisa sinyal crypto menggunakan machine learning ensemble model dengan FastAPI.

---

## Fitur

- Mengambil data OHLCV dari exchange Indodax via `ccxt`
- Menghitung indikator teknikal: EMA50, EMA200, MACD, Bollinger Bands
- Melatih model ensemble (RandomForest, XGBoost, SGD) secara otomatis saat request
- Prediksi sinyal BUY / SELL / HOLD dengan confidence probabilitas
- Frontend minimalis menggunakan HTML + Tailwind CSS
- API dengan FastAPI dan CORS support

---

## Instalasi

1. Clone repo ini

```bash
git clone https://github.com/username/simple-crypto-ai.git
cd simple-crypto-ai


2. Buat virtual environment dan aktifkan


python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


3. Install dependencies

pip install -r requirements.txt


## Menjalankan

uvicorn main:app --reload
Buka browser dan akses: http://localhost:8000

Struktur Folder

.
├── main.py              # Entry point FastAPI app
├── services.py          # Logic ML dan analisa crypto
├── frontend/            # Folder berisi file HTML + CSS frontend
├── requirements.txt     # Daftar dependencies Python
└── test_main.py         # Contoh unit testing API dengan pytest


## Testing
Jalankan unit test dengan:


pytest test_main.py


## Kontribusi
Pull request dan issue sangat diterima!
Pastikan kode mengikuti style dan sudah dites.

Lisensi
MIT License © 2025 Dnayaka