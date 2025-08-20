from fastapi import APIRouter
from loguru import logger

auth_router = APIRouter(prefix="/auth")

@auth_router.get("/")
async def auth_test():
    return {"status": "OAuth auth system working"}
