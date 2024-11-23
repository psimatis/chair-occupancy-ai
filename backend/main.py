import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from api import app

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
