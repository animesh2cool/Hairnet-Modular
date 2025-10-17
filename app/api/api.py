"""
Main API Router
Combines all endpoint routers
"""
from fastapi import APIRouter
from .endpoints import streaming, control, hooter, status

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(streaming.router, tags=["streaming"])
api_router.include_router(control.router, tags=["control"])
api_router.include_router(hooter.router, prefix="/hooter", tags=["hooter"])
api_router.include_router(status.router, tags=["status"])