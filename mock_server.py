from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import random
import time
import argparse

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/inference")
async def inference(request: InferenceRequest):
    # Simulate some processing time
    time.sleep(random.uniform(0.1, 0.5))
    
    return {
        "text": f"Mock response to: {request.prompt}",
        "server_id": app.state.server_id,
        "tokens_used": random.randint(10, request.max_tokens)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "server_id": app.state.server_id}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-id", type=str, required=True)
    args = parser.parse_args()
    
    app.state.server_id = args.server_id
    uvicorn.run(app, host="0.0.0.0", port=args.port)