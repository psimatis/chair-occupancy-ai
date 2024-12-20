import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from api import app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://47.129.230.74"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
