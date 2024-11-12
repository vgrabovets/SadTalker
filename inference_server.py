from typing import Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, ConfigDict

from inference import get_args, main

app = FastAPI()


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driven_audio: str
    source_image: str
    head_motion_scale: float
    pose_style: int
    enhancer: str
    save_path: str


@app.post("/generate")
async def generate(
    requests: List[GenerateRequest], background_tasks: BackgroundTasks
) -> Dict[str, str]:
    for request in requests:
        args = get_args()

        args.driven_audio = request.driven_audio
        args.source_image = request.source_image
        args.head_motion_scale = request.head_motion_scale
        args.pose_style = request.pose_style
        args.enhancer = request.enhancer
        args.save_path = request.save_path

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
