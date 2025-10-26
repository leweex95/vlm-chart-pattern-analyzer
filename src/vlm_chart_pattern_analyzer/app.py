import sys
import asyncio
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import tempfile
from contextlib import asynccontextmanager
from .inference import run_inference
from .models import load_model

# Global model and processor
model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    # Load model on startup
    print("Skipping model loading for local testing")
    model, processor = None, None  # Skip for testing
    yield
    # Cleanup if needed

app = FastAPI(title="VLM Inference API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    print(f"Starting inference for uploaded file: {file.filename}")
    output_dir = Path("./data/results")  # Changed for local testing
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save uploaded file to temp location
    print("Saving uploaded file to temporary location...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)
    
    print(f"File saved to {temp_path}, starting inference...")
    try:
        if model is None or processor is None:
            print("Model not loaded, returning dummy result for testing")
            sys.exit(1)
        else:
            result = await asyncio.to_thread(run_inference, temp_path, model, processor)
        print("Inference completed successfully")
        return {"result": result}
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        temp_path.unlink()  # Clean up temp file
        print("Temporary file cleaned up")

@app.post("/batch_infer")
async def batch_infer(zip_file: UploadFile = File(...)):
    buffer = BytesIO(await zip_file.read())
    results = []
    with ZipFile(buffer) as archive:
        for file_name in archive.namelist():
            with archive.open(file_name) as f:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(f.read())
                    temp_path = Path(temp_file.name)
                try:
                    result = run_inference(temp_path, model, processor)
                    results.append({file_name: result})
                finally:
                    temp_path.unlink()
    return {"results": results}

