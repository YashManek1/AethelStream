"""AethelStream M1 — FastAPI service for on-demand model sharding."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..pipeline.writer import write_shards
from ..pipeline.verifier import verify_index_integrity

app = FastAPI(title="AethelStream Shard Engine", version="1.0.0")

class ShardRequest(BaseModel):
    model_name: str = "gpt2"
    output_dir: str = "model_shards"
    num_workers: int = 4

class ShardResponse(BaseModel):
    status: str
    output_dir: str
    num_params: int
    index_path: str

@app.post("/shard", response_model=ShardResponse)
def shard_model(req: ShardRequest) -> ShardResponse:
    try:
        shard_index, _ = write_shards(req.model_name, req.output_dir, num_proc=req.num_workers)
        return ShardResponse(
            status="success",
            output_dir=req.output_dir,
            num_params=len(shard_index),
            index_path=str(Path(req.output_dir) / "shard_index.json"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/verify")
def verify_shards(output_dir: str) -> dict:
    import json
    index_path = Path(output_dir) / "shard_index.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"shard_index.json not found in {output_dir}")
    with open(index_path) as f:
        shard_index = json.load(f)
    try:
        verify_index_integrity(shard_index, output_dir)
        return {"status": "ok", "num_params": len(shard_index)}
    except AssertionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "module": "M1-shard-engine"}
