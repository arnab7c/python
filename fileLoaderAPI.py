from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
from pathlib import Path
import shutil
from src.Utility import get_base_dir, get_raw_data_directory,setup_logger
import os
import logging
import src.Ingestion as Ingestion
from threading import Event


class FileLoader:
    def __init__(self):
        self.app = FastAPI()
        self.stop_event = Event()

        # Resolve paths safely
        self.base_dir = get_base_dir()
        self.upload_dir = get_raw_data_directory()

        log_file = "FileLoader.log"
        self.logger = setup_logger(log_file)
        self.logger.info("Starting Ingestion module.")
        self.ingestion = Ingestion.Ingestion()

        self._register_routes()

    def _register_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return self._html_page()

        @self.app.post("/stop")
        async def stop_ingestion():
            self.stop_event.set()
            return {"message": "Ingestion stop requested"}

        @self.app.post("/upload")
        async def upload_files(
            background_tasks: BackgroundTasks,
            files: List[UploadFile] = File(...)
        ):
            saved_files = []

            for file in files:
                safe_name = Path(file.filename).name
                extension = Path(safe_name).suffix.lower().replace(".", "")
                destination = os.path.join(self.upload_dir,safe_name)

                with open(destination, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                    # Schedule GenAI ingestion
                    background_tasks.add_task(
                        self.ingestion(safe_name, extension,self.stop_event)
                    )

                saved_files.append(safe_name)

            return JSONResponse(
                content={
                    "message": "Files uploaded successfully",
                    "files": saved_files
                }
            )

    def _html_page(self) -> str:
        return """
<!DOCTYPE html>
<html>
<head>
    <title>GenAI File Upload</title>
    <style>
        body { font-family: Arial; padding: 40px; }
        .file-row { margin-bottom: 12px; }
        progress { width: 350px; height: 18px; }
    </style>
</head>
<body>

<h2>Training Module - RAG</h2>

<input type="file" id="files" multiple />
<br><br>

<button onclick="uploadFiles()">Upload</button>

<div id="fileList"></div>

<p id="status"></p>

<script>
function uploadFiles() {
    const files = document.getElementById("files").files;
    if (!files.length) {
        alert("Please select files");
        return;
    }

    const container = document.getElementById("fileList");
    container.innerHTML = "";

    // Create UI rows
    for (let i = 0; i < files.length; i++) {
        const row = document.createElement("div");
        row.className = "file-row";
        row.innerHTML = `
            <strong>${files[i].name}</strong><br>
            <progress id="progress-${i}" value="0" max="100"></progress>
            <span id="text-${i}">0%</span>
        `;
        container.appendChild(row);
    }

    // Upload each file separately
    for (let i = 0; i < files.length; i++) {
        uploadSingleFile(files[i], i);
    }
}

function uploadSingleFile(file, index) {
    const formData = new FormData();
    formData.append("files", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);

    xhr.upload.onprogress = function (event) {
        if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            document.getElementById(`progress-${index}`).value = percent;
            document.getElementById(`text-${index}`).innerText = percent + "%";
        }
    };

    xhr.onload = function () {
        if (xhr.status === 200) {
            document.getElementById(`text-${index}`).innerText = "Done";
        } else {
            document.getElementById(`text-${index}`).innerText = "Failed";
        }
    };

    xhr.send(formData);
}
</script>

<button onclick="stopIngestion()" style="margin-left:10px;color:red;">
    Quit/Stop
</button>

<script>
function stopIngestion() {
    fetch("/stop", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            document.getElementById("status").innerText =
                "Stop requested. Ingestion will halt safely.";
        });
}
</script>

</body>
</html>
"""


# FastAPI entrypoint
file_loader = FileLoader()
app = file_loader.app

