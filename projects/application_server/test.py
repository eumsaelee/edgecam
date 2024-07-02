import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager

from edgecam.readers import WebsocketReader


URI = 'ws://172.27.1.11:8000/inference/det'
READER = None


async def init():
    global READER
    READER = WebsocketReader()
    await READER.open(URI)

async def get_data():
    count = 0
    while count < 100:
        count += 1
        await READER.read()
        await asyncio.sleep(0)
        print(f'{count}: ok. good.')

async def close():
    READER.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init()
    await get_data()
    yield
    await close()


app = FastAPI(lifespan=lifespan)