from typing import Dict

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from inference import get_args, main

app = FastAPI()


class GenerateRequest(BaseModel):
    driven_audio: str
    source_image: str
    head_motion_scale: float
    pose_style: int
    enhancer: str


@app.post("/generate")
async def generate(
    body: GenerateRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    args = get_args()

    args.driven_audio = body.driven_audio
    args.source_image = body.source_image
    args.head_motion_scale = body.head_motion_scale
    args.pose_style = body.pose_style
    args.enhancer = body.enhancer

    background_tasks.add_task(main, args)

    return {"message": "Task is being processed", "status": "accepted"}


if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host="127.0.0.1",
        port=8005,
        log_level="info",
        # reload=True,
    )
