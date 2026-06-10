from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import shutil
import os
from generateEmb import embeddings
from summarizer import generateResponse
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