import logging
import os
import uuid
from typing import Annotated

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from supervisor_agent import SupervisorAgent
from vector_store_manager import VectorStoreManager

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

supervisor = SupervisorAgent()
vector_store = VectorStoreManager()

UPLOAD_DIR = config.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(request: QueryRequest, req:Request):
    user_id = req.headers.get("X-User-ID") or str("random_user_id-" + str(uuid.uuid4()))
    result = supervisor.send_query(request.query, user_id)
    return {"result": result}


@app.post("/upload", responses={400: {"description": "Invalid file"}})
async def upload(file: Annotated[UploadFile, File()]):
    if file is None or file.filename is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    vector_store.add_documents_to_vector_store(file_path)
    return {"message": f"{file.filename} uploaded and embedded successfully"}

