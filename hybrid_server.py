#!/usr/bin/env python3
"""
Hybrid Server - FastAPI + MCP on Single Port

This server provides:
1. MCP server functionality on /mcp endpoint
2. File download service for trained_models directory  
3. Static file serving for HTML reports
4. Health check and API endpoints

All services run on a single port (8090) with proper MCP integration.
"""

import logging
import asyncio
import threading
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

# Import the MCP server config
from src.mcp_nn_tool.config import BASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRAINED_MODELS_DIR = Path("./trained_models")
REPORTS_DIR = Path("./reports")
MCP_PORT = 8091  # MCP subprocess runs on this port

# Ensure directories exist
TRAINED_MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Global variable to hold MCP process
mcp_process = None


def start_mcp_subprocess():
    """Start MCP server as subprocess on MCP_PORT"""
    global mcp_process
    try:
        cmd = ["python", "simple_mcp_server.py", "--port", str(MCP_PORT), "--host", "0.0.0.0"]
        mcp_process = subprocess.Popen(cmd, cwd=".")
        logger.info(f"MCP server subprocess started on port {MCP_PORT}")
        time.sleep(2)  # Wait for server to start
        return True
    except Exception as e:
        logger.error(f"Failed to start MCP subprocess: {e}")
        return False


def stop_mcp_subprocess():
    """Stop MCP server subprocess"""
    global mcp_process
    if mcp_process:
        logger.info("Terminating MCP server subprocess...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("MCP process did not terminate gracefully, forcing kill.")
            mcp_process.kill()
        logger.info("MCP server subprocess stopped")


# Lifespan manager: handles startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hybrid Server...")
    logger.info("Starting MCP server subprocess...")

    def start_in_thread():
        success = start_mcp_subprocess()
        if success:
            logger.info("MCP server subprocess started successfully")
        else:
            logger.error("Failed to start MCP server subprocess")

    # Start MCP in a background thread to avoid blocking startup
    thread = threading.Thread(target=start_in_thread, daemon=True)
    thread.start()

    yield  # ← Application is now live, ready to handle requests

    # After application shutdown
    logger.info("Shutting down Hybrid Server...")
    stop_mcp_subprocess()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MCP BO Tool Hybrid Server",
    description="Bayesian Optimization MCP Tool with integrated file services",
    version="1.0.0",
    lifespan=lifespan  # Use lifespan instead of on_event
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if TRAINED_MODELS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(TRAINED_MODELS_DIR)), name="static")
    app.mount("/models", StaticFiles(directory=str(TRAINED_MODELS_DIR)), name="models")


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "MCP NN Tool Hybrid Server",
        "version": "1.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "download": "/download/file/{file_path}",
            "health": "/health",
            "static_reports": "/static/",
            "models_browser": "/models/"
        },
        "description": "Neural Network MCP Tool with integrated MCP and file services"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "trained_models_dir": str(TRAINED_MODELS_DIR),
        "models_exist": TRAINED_MODELS_DIR.exists(),
        "reports_dir": str(REPORTS_DIR),
        "reports_exist": REPORTS_DIR.exists(),
        "mcp_status": "integrated",
        "mcp_process_running": mcp_process is not None and mcp_process.poll() is None,
        "mcp_url": f"http://localhost:{MCP_PORT}/mcp"
    }


@app.get("/download/file/{file_path:path}")
async def download_file(file_path: str):
    """
    Download files from trained_models directory
    """
    full_path = TRAINED_MODELS_DIR / file_path

    try:
        resolved_path = full_path.resolve()
        trained_models_resolved = TRAINED_MODELS_DIR.resolve()

        if not str(resolved_path).startswith(str(trained_models_resolved)):
            raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directory")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}")

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not resolved_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")

    return FileResponse(
        path=str(resolved_path),
        filename=resolved_path.name,
        media_type='application/octet-stream'
    )


@app.get("/list/models")
async def list_model_files():
    """List all files in the trained_models directory"""
    if not TRAINED_MODELS_DIR.exists():
        return {"files": [], "message": "Trained models directory does not exist"}

    files = []
    for item in TRAINED_MODELS_DIR.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(TRAINED_MODELS_DIR)
            files.append({
                "path": str(relative_path),
                "name": item.name,
                "size": item.stat().st_size,
                "download_url": f"{BASE_URL}/download/file/{relative_path}"
            })

    return {"files": files, "count": len(files)}


# MCP Proxy
import httpx
from fastapi import Response


@app.api_route("/mcp", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
@app.api_route("/mcp/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def mcp_proxy(request: Request, path: str = ""):
    """
    MCP proxy endpoint - forwards requests to MCP server subprocess
    """
    try:
        body = await request.body()
        headers = dict(request.headers)
        method = request.method

        # Clean headers
        headers.pop('host', None)
        headers.pop('content-length', None)

        # Target URL
        target_url = f"http://localhost:{MCP_PORT}/mcp"
        if path:
            target_url += f"/{path}"

        logger.info(f"Proxying MCP Request: {method} {target_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=target_url,
                content=body,
                headers=headers
            )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )

    except httpx.ConnectError:
        logger.error("Cannot connect to MCP server - is it running?")
        return Response(
            content="MCP server not available",
            status_code=503,
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error in MCP proxy: {str(e)}")
        return Response(
            content=f"MCP proxy error: {str(e)}",
            status_code=500,
            media_type="text/plain"
        )


def main():
    """Main function to start the hybrid server"""
    logger.info("Starting Hybrid Server with integrated MCP and FastAPI...")
    logger.info(f"Trained models directory: {TRAINED_MODELS_DIR.absolute()}")
    logger.info(f"Reports directory: {REPORTS_DIR.absolute()}")
    logger.info("Services available:")
    logger.info("  - MCP server: http://localhost:8090/mcp")
    logger.info("  - File downloads: http://localhost:8090/download/file/{path}")
    logger.info("  - Static reports: http://localhost:8090/static/{path}")
    logger.info("  - API docs: http://localhost:8090/docs")

    # Run the server (pass app instance directly to avoid double import)
    uvicorn.run(
        app,  # ← Direct instance, not string
        host="0.0.0.0",
        port=8090,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()