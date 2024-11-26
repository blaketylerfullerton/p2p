import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

class LLMClient:
    def __init__(self, base_url: str, max_retries: int = 3, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def check_health(self) -> Dict:
        """Check the health status of the load balancer"""
        try:
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Health check failed with status {response.status}")
                    return {"status": "unhealthy", "error": await response.text()}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def inference(self, request: InferenceRequest) -> Dict:
        """Send a single inference request"""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                async with self.session.post(
                    f"{self.base_url}/inference",
                    json=request.to_dict(),
                    timeout=self.timeout
                ) as response:
                    self.total_requests += 1
                    if response.status == 200:
                        result = await response.json()
                        latency = time.time() - start_time
                        self.successful_requests += 1
                        self.total_latency += latency
                        logger.info(f"Request successful. Latency: {latency:.2f}s")
                        return result
                    else:
                        logger.warning(f"Request failed with status {response.status}")
                        self.failed_requests += 1
                        if attempt == self.max_retries - 1:
                            raise Exception(f"Failed after {self.max_retries} attempts")
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.failed_requests += 1
                    raise

    async def batch_inference(self, requests: List[InferenceRequest], 
                            max_concurrent: int = 5) -> List[Dict]:
        """Send multiple inference requests concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        async def bounded_inference(request):
            async with semaphore:
                return await self.inference(request)
        
        return await asyncio.gather(
            *[bounded_inference(req) for req in requests],
            return_exceptions=True
        )

    def get_stats(self) -> Dict:
        """Get client statistics"""
        avg_latency = self.total_latency / self.successful_requests if self.successful_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_latency": avg_latency,
            "success_rate": (self.successful_requests / self.total_requests * 100) 
                if self.total_requests > 0 else 0
        }

async def main():
    parser = argparse.ArgumentParser(description='LLM Client')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Load balancer URL')
    parser.add_argument('--prompts', type=str, nargs='+', 
                       default=['Hello, how are you?'],
                       help='List of prompts to process')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum number of concurrent requests')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens for response')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for response generation')
    
    args = parser.parse_args()

    async with LLMClient(args.url) as client:
        # Check health first
        health_status = await client.check_health()
        logger.info(f"Load balancer health: {health_status}")

        if health_status.get('status') != 'healthy':
            logger.error("Load balancer is not healthy. Exiting.")
            return

        # Create request objects
        requests = [
            InferenceRequest(
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            for prompt in args.prompts
        ]

        # Send requests
        try:
            results = await client.batch_inference(
                requests,
                max_concurrent=args.max_concurrent
            )
            
            # Print results
            for prompt, result in zip(args.prompts, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process prompt '{prompt}': {str(result)}")
                else:
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Response: {json.dumps(result, indent=2)}")

            # Print statistics
            stats = client.get_stats()
            logger.info("Client Statistics:")
            logger.info(json.dumps(stats, indent=2))

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())