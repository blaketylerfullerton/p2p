from fastapi import FastAPI, HTTPException
import httpx
import asyncio
from typing import List, Dict, Optional
import random
from pydantic import BaseModel
import logging
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class ServerConfig(BaseModel):
    url: str
    weight: int = 1
    healthy: bool = True
    last_response_time: float = 0
    failed_attempts: int = 0

class LoadBalancer:
    def __init__(self, servers: List[Dict[str, str]], strategy: str = "round_robin"):
        self.servers = {
            server["url"]: ServerConfig(
                url=server["url"],
                weight=server.get("weight", 1)
            ) for server in servers
        }
        self.strategy = strategy
        self.current_idx = 0
        self.last_health_check = time.time()
        self.health_check_interval = 30  # seconds
        self.max_retries = 3
        self.request_queue = asyncio.Queue()
        self.processing = True

    async def health_check(self, client: httpx.AsyncClient, server_url: str) -> bool:
        try:
            response = await client.get(f"{server_url}/health", timeout=5.0)
            healthy = response.status_code == 200
            self.servers[server_url].healthy = healthy
            self.servers[server_url].failed_attempts = 0
            return healthy
        except Exception as e:
            logger.error(f"Health check failed for {server_url}: {str(e)}")
            self.servers[server_url].failed_attempts += 1
            if self.servers[server_url].failed_attempts >= self.max_retries:
                self.servers[server_url].healthy = False
            return False

    async def periodic_health_checks(self):
        async with httpx.AsyncClient() as client:
            while self.processing:
                tasks = []
                for server_url in self.servers:
                    tasks.append(self.health_check(client, server_url))
                await asyncio.gather(*tasks)
                await asyncio.sleep(self.health_check_interval)

    def get_next_server(self) -> Optional[str]:
        healthy_servers = [url for url, config in self.servers.items() if config.healthy]
        if not healthy_servers:
            return None

        if self.strategy == "round_robin":
            server = healthy_servers[self.current_idx % len(healthy_servers)]
            self.current_idx += 1
            return server
        elif self.strategy == "least_response_time":
            return min(
                healthy_servers,
                key=lambda x: self.servers[x].last_response_time
            )
        elif self.strategy == "weighted_random":
            weights = [self.servers[server].weight for server in healthy_servers]
            return random.choices(healthy_servers, weights=weights)[0]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def handle_request(self, request: InferenceRequest) -> Dict:
        server_url = self.get_next_server()
        if not server_url:
            raise HTTPException(status_code=503, detail="No healthy servers available")

        start_time = time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{server_url}/inference",
                    json=request.dict(),
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Update server stats
                self.servers[server_url].last_response_time = time.time() - start_time
                return response.json()
                
            except Exception as e:
                logger.error(f"Request to {server_url} failed: {str(e)}")
                self.servers[server_url].failed_attempts += 1
                if self.servers[server_url].failed_attempts >= self.max_retries:
                    self.servers[server_url].healthy = False
                raise HTTPException(status_code=503, detail=str(e))

app = FastAPI()
load_balancer = LoadBalancer([
    {"url": "http://localhost:8001"},
    {"url": "http://localhost:8002"},
    {"url": "http://localhost:8003"}
], strategy="round_robin")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_balancer.periodic_health_checks())

@app.on_event("shutdown")
async def shutdown_event():
    load_balancer.processing = False

@app.post("/inference")
async def inference(request: InferenceRequest):
    return await load_balancer.handle_request(request)

@app.get("/health")
async def health():
    healthy_servers = sum(1 for server in load_balancer.servers.values() if server.healthy)
    if healthy_servers == 0:
        raise HTTPException(status_code=503, detail="No healthy backend servers")
    return {
        "status": "healthy",
        "healthy_servers": healthy_servers,
        "total_servers": len(load_balancer.servers)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)