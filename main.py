from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import openai
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Contextual Chat API", description="Chat API with GPT-4o-mini integration")

# In-memory storage for chat sessions
chat_sessions: Dict[str, List[Dict]] = {}

# Pydantic models for request/response
class ChatCreateResponse(BaseModel):
    chat_id: str
    message: str

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str
    chat_id: str

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/create_chat", response_model=ChatCreateResponse)
async def create_chat():
    """Create a new chat session and return the chat ID"""
    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = []
    
    return ChatCreateResponse(
        chat_id=chat_id,
        message=f"Chat created successfully with ID: {chat_id}"
    )

@app.post("/chat/{chat_id}/message", response_model=MessageResponse)
async def send_message(chat_id: str, request: MessageRequest):
    """Send a message to a specific chat and get GPT-4o-mini response"""
    
    # Check if chat exists
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    }
    chat_sessions[chat_id].append(user_message)
    
    try:
        # Prepare messages for OpenAI API
        messages = [{"role": msg["role"], "content": msg["content"]} 
                   for msg in chat_sessions[chat_id] if msg["role"] in ["user", "assistant"]]
        
        # Call OpenAI GPT-4o-mini
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[chat_id].append(assistant_message)
        
        return MessageResponse(
            response=assistant_response,
            chat_id=chat_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str):
    """Get the full chat history for a specific chat ID"""
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return {
        "chat_id": chat_id,
        "history": chat_sessions[chat_id]
    }

@app.get("/chats")
async def list_chats():
    """List all active chat sessions"""
    return {
        "active_chats": list(chat_sessions.keys()),
        "total_chats": len(chat_sessions)
    }

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a specific chat session"""
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    del chat_sessions[chat_id]
    return {"message": f"Chat {chat_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
