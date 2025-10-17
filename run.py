"""
Application Startup Script
Runs the FastAPI application with Uvicorn
"""
import uvicorn
from app.core.config import ServerConfig

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=ServerConfig.RELOAD,
        log_level="info"
    )