from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from communication import communication_bus
from training_manager import TrainingManager
from user_response_generator import user_response_generator
from SystemMonitor import system_monitor
import uvicorn

# Import route modules
from api_endpoints import router as api_router

# Import TaskManager
from TaskManager import TaskManager

app = FastAPI(title="OctoPusAI", version="1.0", description="AI model training and evaluation platform")

# Initialize managers
task_manager = TaskManager()
training_manager = TrainingManager()

# Setup CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This can be set to more restrictive origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from other modules
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to OctoPusAI API!"}

# Include any startup or shutdown events
@app.on_event("startup")
async def startup_event():
    system_monitor.log_activity("Application startup")
    # Example: Initialize MLflow, database connections, etc.
    print("Application startup")
    # Initialize or load any necessary resources for TaskManager
    await task_manager.initialize_resources()

@app.on_event("shutdown")
async def shutdown_event():
    system_monitor.log_activity("Application shutdown")
    # Example: Close database connections
    print("Application shutdown")
    await task_manager.shutdown_resources()
    
@app.get("/process_prompt")
async def process_prompt(prompt: str):
    # Process the prompt through TaskManager
    # This is simplified; actual implementation may vary
    task = {'prompt': prompt, 'task': 'text_summarization'} # Example
    await task_manager.queue_task(task)
    # Response generation is handled asynchronously through communication bus

    return {"message": "Your request is being processed."}
