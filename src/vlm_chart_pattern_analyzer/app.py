import asyncio
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from .inference import run_inference


app = FastAPI(title="VLM Inference API")


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    output_dir = Path("/app/data/results")
    output_dir.mkdir(exist_ok=True)
    result = await asyncio.to_thread(run_inference, file.file, output_dir)
    return {"result": result}

@app.post("/batch_infer")
async def batch_infer(zip_file: UploadFile = File(...)):
    buffer = BytesIO(await zip_file.read())
    results = []
    with ZipFile(buffer) as archive:
        for file_name in archive.namelist():
            with archive.open(file_name) as f:
                results.append(run_inference(f, Path("/app/data/results")))
    return {"results": results}

