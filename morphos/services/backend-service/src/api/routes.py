from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

router = APIRouter()


@router.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "service": "Morphos Backend Service",
        "status": "running",
        "version": "0.1.0",
    }
