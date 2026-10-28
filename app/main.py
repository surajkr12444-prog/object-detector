from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .api.endpoints import router as api_router
from .db import Base, engine

# Initialize SQLite database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="VisionX AI API",
    version="0.1.0",
    description="Backend API for real-time object detection, tracking and analytics using YOLO",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # If client accepts HTML, show a sleek landing page
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(
            """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VisionX AI Backend</title>
  <style>
    body {
      margin: 0;
      background: #0f172a;
      color: #f8fafc;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      padding: 2.5rem;
      border-radius: 1rem;
      max-width: 540px;
      width: 90%;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
      text-align: center;
    }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      margin-bottom: 1rem;
      border: 1px solid rgba(52, 211, 153, 0.3);
    }
    h1 {
      margin: 0 0 0.5rem;
      font-size: 1.85rem;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p {
      color: #94a3b8;
      line-height: 1.6;
      margin: 0 0 1.5rem;
    }
    .buttons {
      display: flex;
      gap: 1rem;
      justify-content: center;
      flex-wrap: wrap;
    }
    a.btn {
      display: inline-flex;
      align-items: center;
      padding: 0.65rem 1.25rem;
      border-radius: 0.5rem;
      text-decoration: none;
      font-weight: 500;
      font-size: 0.95rem;
      transition: all 0.2s;
    }
    a.primary {
      background: #0284c7;
      color: #fff;
    }
    a.primary:hover {
      background: #0369a1;
    }
    a.secondary {
      background: #334155;
      color: #f1f5f9;
    }
    a.secondary:hover {
      background: #475569;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">&#9679; API Online & Ready</div>
    <h1>VisionX AI Backend</h1>
    <p>High-performance FastAPI inference engine powered by Ultralytics YOLOv8 with real-time tracking and analytics.</p>
    <div class="buttons">
      <a class="btn primary" href="http://localhost:5173" target="_blank">Open Frontend Dashboard &rarr;</a>
      <a class="btn secondary" href="/docs">Interactive API Docs</a>
      <a class="btn secondary" href="/health">Health Status</a>
    </div>
  </div>
</body>
</html>"""
        )
    return JSONResponse(
        {
            "status": "online",
            "service": "visionx-backend",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "frontend": "http://localhost:5173",
        }
    )


# Mount all endpoints
app.include_router(api_router)
