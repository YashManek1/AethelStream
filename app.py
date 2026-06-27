from fastapi import FastAPI
from pydantic import BaseModel
from sharding.shard import shard_model

app = FastAPI()


class ModelRequest(BaseModel):
    model_name: str = "gpt2"


@app.post("/shard-model")
def shard(req: ModelRequest):
    result = shard_model(req.model_name)
    return result


@app.get("/")
def home():
    return {"message": "AethelStream API running"}