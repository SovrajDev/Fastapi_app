from fastapi import FastAPI, Request, Query, Depends,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from newapp import *
from pydantic import BaseModel
import os
import google.generativeai as genai
#from dotenv import load_dotenv
import httpx
app = FastAPI()


# Enable CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],allow_credentials=True)

# Routes

@app.get("/")
def main_home():
    return {"message": "Welcome"}

@app.get("/api")
def home():
    return {
        "Page": "Home",
        "_AvailableRoutes": {
            "hybrid_recommendation": "/hybrid_recommendation?book_title=BookName",
            "content_recommendation": "/content_recommendation?book_title=BookName",
            "collaborative_recommendation": "/collaborative_recommendation?book_title=BookName",
        },
    }

@app.get("/api/top_books")
def top_books():
    return {"data": df_popularity.to_dict(orient="records")}

@app.get("/api/recommend")
def recommendation(book_title: str = Query(..., description="Title of the book")):
    recommendations = recommend(book_title)
    return recommendations

@app.get("/api/content_recommendation")
def content_recommendation(book_title: str = Query(..., description="Title of the book")):
    recommendations = content_based_recommender(book_title, df_books)
    return recommendations

@app.get("/api/collaborative_recommendation")
def collaborative_recommendation(book_title: str = Query(..., description="Title of the book")):
    recommendations = recommend(book_title)
    return recommendations

@app.get("/api/hybrid_recommendation")
def hybrid_recommendation_route(book_title: str = Query(..., description="Title of the book")):
    recommendations = hybrid_recommendation(book_title,df_books)
    return recommendations
#OPENAI_API_KEY = "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"
#load_dotenv()
Gemini_api_key='AIzaSyBojxWJS0AKTKIIvQkb2G7ZDpSqVTmOnWY'
os.environ["Gemini_api_key"]=Gemini_api_key
genai.configure(api_key=os.environ["Gemini_api_key"])
class BookRequest(BaseModel):
    book_name: str

@app.post("/summarize")
async def summarize_book(request: BookRequest):
    try:
      # Model Configuration
         model_config = {
           "temperature": 1,
            "top_p": 0.99,
            "top_k": 0,
            "max_output_tokens": 150,
            }

         model = genai.GenerativeModel('gemini-1.5-flash-latest',generation_config=model_config)
         response = model.generate_content(f"Summarize the book{request.book_name}")
       #print(response.text)
         return {"summary": response.text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Error getting book summary from OpenAI")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
