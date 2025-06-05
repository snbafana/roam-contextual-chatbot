from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
import json
from utils import create_game_modification_agent, TEMPLATE_PROMPT
import asyncio
import logging
from pathlib import Path
from agents import Runner, Agent

load_dotenv()

# Set up logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
        logging.FileHandler(LOGS_DIR / "app.log")  # Add file handler for all logs
    ]
)

# Configure logging
def setup_logger(chat_id: str) -> logging.Logger:
    """Set up a logger for a specific chat session"""
    logger = logging.getLogger(f"chat_{chat_id}")
    logger.setLevel(logging.INFO)
    
    # Create a file handler for this chat session
    log_file = LOGS_DIR / f"chat_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger

app = FastAPI(title="Contextual Chat API", description="Chat API with GPT-4o-mini integration")

# In-memory storage for chat sessions
chat_sessions: Dict[str, List[Dict]] = {}
chat_loggers: Dict[str, logging.Logger] = {}
chat_agents: Dict[str, Agent] = {}

# Add startup event to log application start
@app.on_event("startup")
async def startup_event():
    logging.info("Application starting up...")
    logging.info(f"Logs directory: {LOGS_DIR.absolute()}")

# Add shutdown event to log application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutting down...")
    # Close all loggers
    for logger in chat_loggers.values():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    chat_loggers.clear()

# Pydantic models for request/response
class ChatCreateResponse(BaseModel):
    chat_id: str
    message: str

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str
    chat_id: str

@app.post("/create_chat", response_model=ChatCreateResponse)
async def create_chat():
    """Create a new chat session and return the chat ID"""
    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = [
        {
            "role": "system",
            "content": TEMPLATE_PROMPT
        }
    ]
    
    # Set up logger for this chat session
    chat_loggers[chat_id] = setup_logger(chat_id)
    logging.info(f"Created new chat session with ID: {chat_id}")
    
    # Create agent for this chat session
    chat_agents[chat_id] = create_game_modification_agent()
    logging.info(f"Created new agent for chat session: {chat_id}")
    
    return ChatCreateResponse(
        chat_id=chat_id,
        message=f"Chat created successfully with ID: {chat_id}"
    )

@app.post("/chat/{chat_id}/message", response_model=MessageResponse)
async def send_message(chat_id: str, request: MessageRequest):
    """Send a message to a specific chat and get GPT-4o-mini response"""
    
    # Check if chat exists
    if chat_id not in chat_sessions:
        logging.error(f"Chat session not found: {chat_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    }
    chat_sessions[chat_id].append(user_message)
    
    # Log user message
    logger = chat_loggers.get(chat_id)
    if logger:
        logger.info(f"User Message: {request.message}")
        logging.info(f"[Chat {chat_id}] User Message: {request.message}")
    
    try:
        # Get the agent for this chat session
        agent = chat_agents[chat_id]
        
        # Track tool call sequence
        tool_call_count = 0
        
        # Run the agent with the user's message
        result = await Runner.run(agent, request.message)
        
        # Log any tool calls and their results
        if logger and hasattr(result, 'tool_calls'):
            logger.info(f"Agent completed {len(result.tool_calls)} tool calls in sequence:")
            logging.info(f"[Chat {chat_id}] Agent completed {len(result.tool_calls)} tool calls in sequence:")
            for i, tool_call in enumerate(result.tool_calls, 1):
                tool_call_msg = f"Tool Call {i}: {tool_call.name} with args: {tool_call.arguments}"
                logger.info(tool_call_msg)
                logging.info(f"[Chat {chat_id}] {tool_call_msg}")
                
                if hasattr(tool_call, 'result'):
                    result_msg = f"Tool Result {i}: {tool_call.result}"
                    logger.info(result_msg)
                    logging.info(f"[Chat {chat_id}] {result_msg}")
                logger.info("-" * 50)
                logging.info(f"[Chat {chat_id}] {'-' * 50}")
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": result.final_output,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[chat_id].append(assistant_message)
        
        # Log assistant response
        if logger:
            logger.info(f"Assistant Response: {result.final_output}")
            logging.info(f"[Chat {chat_id}] Assistant Response: {result.final_output}")
        
        return MessageResponse(
            response=result.final_output,
            chat_id=chat_id
        )
        
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        if logger:
            logger.error(error_msg)
        logging.error(f"[Chat {chat_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/chat/{chat_id}/message/stream")
async def send_message_stream(chat_id: str, request: MessageRequest):
    """Send a message to a specific chat and get GPT-4o-mini streaming response"""
    
    # Check if chat exists
    if chat_id not in chat_sessions:
        logging.error(f"Chat session not found: {chat_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    }
    chat_sessions[chat_id].append(user_message)
    
    # Get logger for this chat session
    logger = chat_loggers.get(chat_id)
    if logger:
        logger.info(f"User Message: {request.message}")
        logger.info(f"[Chat {chat_id}] User Message: {request.message}")
    
    async def generate_stream():
        try:
            # Get the agent for this chat session
            agent = chat_agents[chat_id]
            
            # Track tool call sequence
            tool_call_count = 0
            
            # Run the agent with streaming
            async for event in Runner.run_streamed(agent, request.message):
                # Handle different event types
                if event.type == "content":
                    # Stream content chunks
                    chunk_data = {
                        "content": event.content,
                        "is_complete": False,
                        "chat_id": chat_id
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                elif event.type == "tool_call":
                    tool_call_count += 1
                    # Log tool call
                    if logger:
                        tool_call_msg = f"Tool Call {tool_call_count}: {event.tool_call.name} with args: {event.tool_call.arguments}"
                        logger.info(tool_call_msg)
                        logging.info(f"[Chat {chat_id}] {tool_call_msg}")
                    
                    # Stream tool call information
                    tool_data = {
                        "tool_call": event.tool_call,
                        "is_complete": False,
                        "chat_id": chat_id,
                        "tool_call_number": tool_call_count
                    }
                    yield f"data: {json.dumps(tool_data)}\n\n"
                
                elif event.type == "tool_result":
                    # Log tool result
                    if logger:
                        result_msg = f"Tool Result {tool_call_count}: {event.tool_result}"
                        logger.info(result_msg)
                        logging.info(f"[Chat {chat_id}] {result_msg}")
                    
                    # Stream tool results
                    result_data = {
                        "tool_result": event.tool_result,
                        "is_complete": False,
                        "chat_id": chat_id,
                        "tool_call_number": tool_call_count
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"
                
                elif event.type == "complete":
                    # Log completion with tool call summary
                    if logger:
                        completion_msg = f"Agent completed {tool_call_count} tool calls in sequence"
                        logger.info(completion_msg)
                        logging.info(f"[Chat {chat_id}] {completion_msg}")
                        
                        final_msg = f"Final Response: {event.final_output}"
                        logger.info(final_msg)
                        logging.info(f"[Chat {chat_id}] {final_msg}")
                    
                    # Stream completion signal
                    completion_data = {
                        "content": "",
                        "is_complete": True,
                        "chat_id": chat_id,
                        "full_response": event.final_output,
                        "total_tool_calls": tool_call_count
                    }
                    yield f"data: {json.dumps(completion_data)}\n\n"
                    
                    # Add final response to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": event.final_output,
                        "timestamp": datetime.now().isoformat()
                    }
                    chat_sessions[chat_id].append(assistant_message)
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "is_complete": True,
                "chat_id": chat_id
            }
            if logger:
                logger.error(f"Error in streaming: {str(e)}")
            logging.error(f"[Chat {chat_id}] Error in streaming: {str(e)}")
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

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
    
    # Remove logger
    if chat_id in chat_loggers:
        logger = chat_loggers[chat_id]
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        del chat_loggers[chat_id]
    
    # Remove agent
    if chat_id in chat_agents:
        del chat_agents[chat_id]
    
    del chat_sessions[chat_id]
    return {"message": f"Chat {chat_id} deleted successfully"}

@app.get("/test_logging")
async def test_logging():
    """Test endpoint to verify logging is working"""
    logging.info("Test log message from root logger")
    logging.warning("Test warning message")
    logging.error("Test error message")
    
    # Create a test chat session
    chat_id = str(uuid.uuid4())
    chat_loggers[chat_id] = setup_logger(chat_id)
    logger = chat_loggers[chat_id]
    
    logger.info("Test log message from chat logger")
    logger.warning("Test warning message from chat logger")
    logger.error("Test error message from chat logger")
    
    return {
        "message": "Logging test completed",
        "chat_id": chat_id,
        "log_files": {
            "app_log": str(LOGS_DIR / "app.log"),
            "chat_log": str(LOGS_DIR / f"chat_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
