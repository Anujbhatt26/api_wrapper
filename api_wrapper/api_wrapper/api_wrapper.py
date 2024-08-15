from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import subprocess
from typing import Optional
import logging
import sys

app = FastAPI()

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


class RunCommandRequest(BaseModel):
    source_path: str = Field(...,
                             description="Path to the source image or video")
    target_path: str = Field(...,
                             description="Path to the target image or video")
    output_path: str = Field(...,
                             description="Path to the output file or directory")
    frame_processor: str = Field(
        default="face_swapper", description="Frame processor to use")
    keep_fps: Optional[bool] = Field(
        default=None, description="Keep the target FPS")
    skip_audio: Optional[bool] = Field(
        default=None, description="Skip target audio")
    many_faces: Optional[bool] = Field(
        default=None, description="Detect many faces")
    output_video_encoder: Optional[str] = Field(
        default=None, description="Output video encoder")
    max_memory: Optional[int] = Field(
        default=None, description="Maximum memory to use (in GB)")
    execution_provider: Optional[str] = Field(
        default=None, description="Execution provider (e.g., cpu, cuda)")
    execution_threads: Optional[int] = Field(
        default=None, description="Number of execution threads")


class ProcessVideoRequest(BaseModel):
    source: str
    target: str
    output: str
    frame_processor: str = "face_swapper"
    keep_fps: bool = False
    keep_frames: bool = False
    skip_audio: bool = False
    many_faces: bool = False
    reference_face_position: Optional[int] = None
    reference_frame_number: Optional[int] = None
    similar_face_distance: Optional[float] = None
    temp_frame_format: str = "jpg"
    temp_frame_quality: int = 100
    output_video_encoder: str = "libx264"
    output_video_quality: int = 100
    max_memory: int = 4
    execution_provider: str = "cpu"
    execution_threads: int = 1


@app.get("/")
async def hello():
    return {"message": "Hello, welcome to the video processing API!"}


@app.post("/run-command/")
async def run_command(request: RunCommandRequest):
    command = [
        "python", "run.py",
        "-s", request.source_path,
        "-t", request.target_path,
        "-o", request.output_path,
        "--frame-processor", request.frame_processor
    ]

    # Adding optional flags and parameters
    if request.keep_fps:
        command.append("--keep-fps")
    if request.skip_audio:
        command.append("--skip-audio")
    if request.many_faces:
        command.append("--many-faces")
    if request.output_video_encoder:
        command.extend(["--output-video-encoder",
                       request.output_video_encoder])
    if request.max_memory:
        command.extend(["--max-memory", str(request.max_memory)])
    if request.execution_provider:
        command.extend(["--execution-provider", request.execution_provider])
    if request.execution_threads:
        command.extend(["--execution-threads", str(request.execution_threads)])

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        stdout_output = []
        stderr_output = []

        for stdout_line in process.stdout:
            print(stdout_line, end='')
            stdout_output.append(stdout_line)

        for stderr_line in process.stderr:
            print(stderr_line, end='', file=sys.stderr)
            stderr_output.append(stderr_line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        logger.debug(stdout_output)

        return {
            "status": "success",
            "stdout": ''.join(stdout_output),
            "stderr": ''.join(stderr_output),
            "test": "test"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "message": "Command execution failed",
            "stdout": ''.join(stdout_output),
            "stderr": ''.join(stderr_output)
        })


@app.post("/process-video/")
async def process_video(request: ProcessVideoRequest):
    # Simulate processing with the received arguments
    return {"Arguments received": request.dict()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
