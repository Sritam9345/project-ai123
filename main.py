from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import shutil
import os
from generateEmb import embeddings
from summarizer import generateResponse
from summarizerv2 import generateResponse as generateResponsev2
from pydantic import BaseModel

app = FastAPI()

class ChunkRequest(BaseModel):
    chunks: list[str]


@app.post("/upload-pdf")
async def upload_pdf(
    pdf: UploadFile = File(...)
):
    
    response = await generateResponse(pdf)
    
    return response
    

@app.post('/generate')
def generateEmbeddings(chunks:ChunkRequest):
    
    
    
    embeds = embeddings(chunks.chunks)
    
    return {
        "embeddings": embeds.tolist()
    }

@app.post('/upload-pdf-v2')
async def upload_pdf(
    pdf: UploadFile = File(...)
):
    
    response = await generateResponsev2(pdf)
    
    return response