from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi import File, UploadFile, Form

from typing import List, Optional, Tuple
import gradio as gr

import os
import shutil
import time


def lama_api(_: gr.Blocks, app: FastAPI):
    @app.post('/lamaRemove/image')
    async def lama_remove(
    ):
        return {
            "server_process_time": "arif ahmed"
        }


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(lama_api)

except:
    pass
