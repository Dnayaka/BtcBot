# 1. Gunakan base image Python
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy file ke container
COPY . .

# 4. Install dependency
RUN pip install --no-cache-dir -r requirements.txt

# 5. Jalankan bot
CMD ["python", "main.py"]
