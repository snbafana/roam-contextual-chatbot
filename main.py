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
from utils import TOOL_SCHEMAS, TEMPLATE_PROMPT, FUNCTION_MAP
import asyncio
import logging
from pathlib import Path

load_dotenv()

# Set up logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

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
    
    return logger

app = FastAPI(title="Contextual Chat API", description="Chat API with GPT-4o-mini integration")

# In-memory storage for chat sessions
chat_sessions: Dict[str, List[Dict]] = {}
chat_loggers: Dict[str, logging.Logger] = {}

async def execute_tool_call(tool_call: Dict, chat_id: str) -> Dict:
    """Execute a tool call and return its result"""
    try:
        func_name = tool_call.get("function", {}).get("name")
        if not func_name or func_name not in FUNCTION_MAP:
            return {"error": f"Unknown function: {func_name}"}
        
        # Get function and arguments
        func = FUNCTION_MAP[func_name]
        args_str = tool_call.get("function", {}).get("arguments", "{}")
        
        try:
            # Parse arguments
            args = json.loads(args_str)
            
            # Log the tool call
            logger = chat_loggers.get(chat_id)
            if logger:
                logger.info(f"Tool Call: {func_name}")
                logger.info(f"Arguments: {json.dumps(args, indent=2)}")
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            
            # Log the result
            if logger:
                logger.info(f"Result: {json.dumps(result, indent=2)}")
            
            return {
                "function": func_name,
                "result": result
            }
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON arguments for {func_name}"
            if logger:
                logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error executing {func_name}: {str(e)}"
            if logger:
                logger.error(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error processing tool call: {str(e)}"
        if logger:
            logger.error(error_msg)
        return {"error": error_msg}

async def format_tool_calls(tool_calls: List[Dict], chat_id: str) -> str:
    """Format tool calls into a readable string"""
    if not tool_calls:
        return ""
    
    formatted_calls = []
    for call in tool_calls:
        try:
            # Extract function name and arguments
            func_name = call.get("function", {}).get("name", "unknown")
            args_str = call.get("function", {}).get("arguments", "{}")
            
            # Try to parse and format arguments if they're JSON
            try:
                args = json.loads(args_str)
                formatted_args = json.dumps(args, indent=2)
            except json.JSONDecodeError:
                formatted_args = args_str
            
            # Execute the function and get result
            result = await execute_tool_call(call, chat_id)
            
            # Create formatted call string
            call_str = f"Function: {func_name}\nArguments:\n{formatted_args}\nResult:\n{json.dumps(result, indent=2)}"
            formatted_calls.append(call_str)
        except Exception as e:
            formatted_calls.append(f"Error formatting tool call: {str(e)}")
    
    return "\n\n".join(formatted_calls)

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
    chat_sessions[chat_id] = [
        {
            "role": "system",
            "content": TEMPLATE_PROMPT
        }
    ]
    
    # Set up logger for this chat session
    chat_loggers[chat_id] = setup_logger(chat_id)
    
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
    
    # Log user message
    logger = chat_loggers.get(chat_id)
    if logger:
        logger.info(f"User Message: {request.message}")
    
    try:
        # Prepare messages for OpenAI API
        messages = [{"role": msg["role"], "content": msg["content"]} 
                   for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
        
        # Call OpenAI GPT-4o-mini with tools
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[{"type": "function", "function": schema} for schema in TOOL_SCHEMAS.values()],
            tool_choice="auto",
            max_tokens=1000,
            temperature=0.7
        )
        
        # Process tool calls if any
        message = response.choices[0].message
        if message.tool_calls:
            # Execute each tool call and collect results
            tool_results = []
            for tool_call in message.tool_calls:
                result = await execute_tool_call(tool_call.dict(), chat_id)
                tool_results.append(result)
            
            # Format tool calls and their results for internal context
            tool_calls_str = await format_tool_calls([call.dict() for call in message.tool_calls], chat_id)
            
            # Add tool results to the response for context
            if tool_results:
                tool_calls_str += "\n\nResults:\n" + json.dumps(tool_results, indent=2)
            
            # Store tool calls in chat history
            tool_message = {
                "role": "assistant",
                "content": message.content + "\n\n" + tool_calls_str,
                "timestamp": datetime.now().isoformat()
            }
            chat_sessions[chat_id].append(tool_message)
            
            # Add function messages with names to follow_up_messages
            follow_up_messages = []
            for tool_call in message.tool_calls:
                follow_up_messages.append({
                    "role": "function",
                    "name": tool_call.function.name,
                    "content": tool_call.function.arguments
                })

            # If this was a find_attributes call, analyze the results
            if message.tool_calls and message.tool_calls[0].function.name == "find_attributes":
                try:
                    # Get the results from the function call
                    results = json.loads(message.tool_calls[0].function.arguments)
                    print(results)
                    
                    # Create a prompt for analyzing the results
                    analysis_prompt = f"""Given these search results from the find_attributes function call:
{json.dumps(results, indent=2)}

Go over the top 3 results, relevant to the user's query,and explain their attributes and subvariables. Only around 5 sentences tops. Lead the user to the next step"""

                    # Prepare messages including full chat history
                    analysis_messages = [{"role": msg["role"], "content": msg["content"]} 
                                       for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
                    analysis_messages.append({"role": "user", "content": analysis_prompt})

                    # Get the analysis
                    analysis_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=analysis_messages
                    )
                    
                    analysis_content = analysis_response.choices[0].message.content
                    
                    # Add the analysis to the follow-up messages
                    follow_up_messages.append({
                        "role": "assistant",
                        "content": analysis_content
                    })
                    
                    # Return the analysis instead of tool results
                    return MessageResponse(
                        response=analysis_content,
                        chat_id=chat_id
                    )
                except Exception as e:
                    print(f"Error analyzing search results: {e}")
            # If this was an edit_attribute call, show the changes
            elif message.tool_calls and message.tool_calls[0].function.name == "edit_attribute":
                try:
                    # Get the results from the function call
                    results = json.loads(message.tool_calls[0].function.arguments)
                    
                    # Create a prompt for showing the changes
                    analysis_prompt = f"""Given these changes from the edit_attribute function call:
{json.dumps(results, indent=2)}

Show the Unity JSON changes in a code block and briefly explain what was modified. Keep it to 2-3 sentences."""

                    # Prepare messages including full chat history
                    analysis_messages = [{"role": msg["role"], "content": msg["content"]} 
                                       for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
                    analysis_messages.append({"role": "user", "content": analysis_prompt})

                    # Get the analysis
                    analysis_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=analysis_messages
                    )
                    
                    analysis_content = analysis_response.choices[0].message.content
                    
                    # Add the analysis to the follow-up messages
                    follow_up_messages.append({
                        "role": "assistant",
                        "content": analysis_content
                    })
                    
                    # Return the analysis instead of tool results
                    return MessageResponse(
                        response=analysis_content,
                        chat_id=chat_id
                    )
                except Exception as e:
                    print(f"Error analyzing edit results: {e}")
            # If this was a create_ui call, show the UI configuration
            elif message.tool_calls and message.tool_calls[0].function.name == "create_ui":
                try:
                    # Get the results from the function call
                    results = json.loads(message.tool_calls[0].function.arguments)
                    
                    # Create a prompt for showing the UI configuration
                    analysis_prompt = f"""Given this UI configuration from the create_ui function call:
{json.dumps(results, indent=2)}

Show the iOS JSON configuration in a code block and briefly describe what UI elements were created. Keep it to 2-3 sentences."""

                    # Prepare messages including full chat history
                    analysis_messages = [{"role": msg["role"], "content": msg["content"]} 
                                       for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
                    analysis_messages.append({"role": "user", "content": analysis_prompt})

                    # Get the analysis
                    analysis_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=analysis_messages
                    )
                    
                    analysis_content = analysis_response.choices[0].message.content
                    
                    # Add the analysis to the follow-up messages
                    follow_up_messages.append({
                        "role": "assistant",
                        "content": analysis_content
                    })
                    
                    # Return the analysis instead of tool results
                    return MessageResponse(
                        response=analysis_content,
                        chat_id=chat_id
                    )
                except Exception as e:
                    print(f"Error analyzing UI results: {e}")
            
            # Return the tool call results directly if not a handled tool call
            return MessageResponse(
                response=tool_calls_str,
                chat_id=chat_id
            )
        else:
            assistant_response = message.content
            
            # Log assistant response
            if logger:
                logger.info(f"Assistant Response: {assistant_response}")
        
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
        error_msg = f"Error calling OpenAI API: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/chat/{chat_id}/message/stream")
async def send_message_stream(chat_id: str, request: MessageRequest):
    """Send a message to a specific chat and get GPT-4o-mini streaming response"""
    
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
    
    async def generate_stream():
        try:
            # Prepare messages for OpenAI API
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
            
            # Call OpenAI GPT-4o-mini with streaming and tools
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[{"type": "function", "function": schema} for schema in TOOL_SCHEMAS.values()],
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            
            full_response = ""
            natural_response = ""
            current_tool_call = {
                "function": {"name": None, "arguments": ""},
                "type": "function"
            }
            tool_calls = []
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    natural_response += content
                    
                    # Send each chunk as JSON
                    chunk_data = {
                        "content": content,
                        "is_complete": False,
                        "chat_id": chat_id
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Handle tool calls
                if chunk.choices[0].delta.tool_calls:
                    delta = chunk.choices[0].delta.tool_calls[0]
                    
                    # If we have a new function name, start a new tool call
                    if delta.function.name:
                        if current_tool_call["function"]["name"]:
                            # Save previous tool call if it exists
                            tool_calls.append(current_tool_call)
                        current_tool_call = {
                            "function": {"name": delta.function.name, "arguments": ""},
                            "type": "function"
                        }
                    
                    # Accumulate arguments
                    if delta.function.arguments:
                        current_tool_call["function"]["arguments"] += delta.function.arguments
            
            # Add the last tool call if it exists
            if current_tool_call["function"]["name"]:
                tool_calls.append(current_tool_call)
            
            # If there were tool calls, execute them and format results
            if tool_calls:
                # Execute each tool call and collect results
                tool_results = []
                for call in tool_calls:
                    result = await execute_tool_call(call, chat_id)
                    tool_results.append(result)
                
                # Format tool calls and their results for internal context
                formatted_calls = []
                for call, result in zip(tool_calls, tool_results):
                    try:
                        # Parse and format arguments if they're JSON
                        args = json.loads(call["function"]["arguments"])
                        formatted_args = json.dumps(args, indent=2)
                        call_str = f"Function: {call['function']['name']}\nArguments:\n{formatted_args}\nResult:\n{json.dumps(result, indent=2)}"
                        formatted_calls.append(call_str)
                    except json.JSONDecodeError:
                        # If not valid JSON, show raw arguments
                        call_str = f"Function: {call['function']['name']}\nArguments: {call['function']['arguments']}\nResult:\n{json.dumps(result, indent=2)}"
                        formatted_calls.append(call_str)
                
                tool_calls_str = "\n\n".join(formatted_calls)
                full_response += f"\n\nTool calls executed:\n{tool_calls_str}"
                
                # Store tool calls in chat history
                tool_message = {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                }
                chat_sessions[chat_id].append(tool_message)
                
                # Add function messages with names to follow_up_messages
                follow_up_messages = []
                for call in tool_calls:
                    follow_up_messages.append({
                        "role": "function",
                        "name": call["function"]["name"],
                        "content": call["function"]["arguments"]
                    })

                # If this was a find_attributes call, analyze the results
                if tool_calls and tool_calls[0]["function"]["name"] == "find_attributes":
                    try:
                        # Get the results from the function call
                        results = json.loads(tool_calls[0]["function"]["arguments"])
                        
                        # Create a prompt for analyzing the results
                        analysis_prompt = f"""Given these search results from the find_attributes function call:
{json.dumps(results, indent=2)}

Go over the top 3 results, that are LISTED, relevant to the user's query, and explain their attributes and subvariables. Only around 5 sentences tops. Lead the user to the next step"""

                        # Prepare messages including full chat history
                        analysis_messages = [{"role": msg["role"], "content": msg["content"]} 
                                           for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
                        analysis_messages.append({"role": "user", "content": analysis_prompt})

                        # Get the analysis
                        analysis_response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=analysis_messages,
                            stream=True
                        )
                        
                        analysis_content = ""
                        for chunk in analysis_response:
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                analysis_content += content
                                
                                # Stream each chunk as JSON
                                chunk_data = {
                                    "content": content,
                                    "is_complete": False,
                                    "chat_id": chat_id
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # Add the analysis to the follow-up messages
                        follow_up_messages.append({
                            "role": "assistant",
                            "content": analysis_content
                        })
                        
                        # Send completion signal with analysis
                        completion_data = {
                            "content": "",
                            "is_complete": True,
                            "chat_id": chat_id,
                            "full_response": analysis_content
                        }
                        yield f"data: {json.dumps(completion_data)}\n\n"
                        
                        # Add analysis to chat history
                        assistant_message = {
                            "role": "assistant",
                            "content": analysis_content,
                            "timestamp": datetime.now().isoformat()
                        }
                        chat_sessions[chat_id].append(assistant_message)
                        return
                    except Exception as e:
                        print(f"Error analyzing search results: {e}")
                # If this was an edit_attribute call, show the changes
                elif tool_calls and tool_calls[0]["function"]["name"] == "edit_attribute":
                    try:
                        # Get the results from the function call
                        results = json.loads(tool_calls[0]["function"]["arguments"])
                        
                        # Create a prompt for showing the changes
                        analysis_prompt = f"""Given these changes from the edit_attribute function call:
{json.dumps(results, indent=2)}

Show the Unity JSON changes in a code block and briefly explain what was modified. Keep it to 2-3 sentences."""

                        # Prepare messages including full chat history
                        analysis_messages = [{"role": msg["role"], "content": msg["content"]} 
                                           for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
                        analysis_messages.append({"role": "user", "content": analysis_prompt})

                        # Get the analysis
                        analysis_response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=analysis_messages,
                            stream=True
                        )
                        
                        analysis_content = ""
                        for chunk in analysis_response:
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                analysis_content += content
                                
                                # Stream each chunk as JSON
                                chunk_data = {
                                    "content": content,
                                    "is_complete": False,
                                    "chat_id": chat_id
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # Add the analysis to the follow-up messages
                        follow_up_messages.append({
                            "role": "assistant",
                            "content": analysis_content
                        })
                        
                        # Send completion signal with analysis
                        completion_data = {
                            "content": "",
                            "is_complete": True,
                            "chat_id": chat_id,
                            "full_response": analysis_content
                        }
                        yield f"data: {json.dumps(completion_data)}\n\n"
                        
                        # Add analysis to chat history
                        assistant_message = {
                            "role": "assistant",
                            "content": analysis_content,
                            "timestamp": datetime.now().isoformat()
                        }
                        chat_sessions[chat_id].append(assistant_message)
                        return
                    except Exception as e:
                        print(f"Error analyzing edit results: {e}")
                # If this was a create_ui call, show the UI configuration
                elif tool_calls and tool_calls[0]["function"]["name"] == "create_ui":
                    try:
                        # Get the results from the function call
                        results = json.loads(tool_calls[0]["function"]["arguments"])
                        
                        # Create a prompt for showing the UI configuration
                        analysis_prompt = f"""Given this UI configuration from the create_ui function call:
{json.dumps(results, indent=2)}

Show the iOS JSON configuration in a code block and briefly describe what UI elements were created. Keep it to 2-3 sentences."""

                        # Prepare messages including full chat history
                        analysis_messages = [{"role": msg["role"], "content": msg["content"]} 
                                           for msg in chat_sessions[chat_id] if msg["role"] in ["system", "user", "assistant"]]
                        analysis_messages.append({"role": "user", "content": analysis_prompt})

                        # Get the analysis
                        analysis_response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=analysis_messages,
                            stream=True
                        )
                        
                        analysis_content = ""
                        for chunk in analysis_response:
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                analysis_content += content
                                
                                # Stream each chunk as JSON
                                chunk_data = {
                                    "content": content,
                                    "is_complete": False,
                                    "chat_id": chat_id
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # Add the analysis to the follow-up messages
                        follow_up_messages.append({
                            "role": "assistant",
                            "content": analysis_content
                        })
                        
                        # Send completion signal with analysis
                        completion_data = {
                            "content": "",
                            "is_complete": True,
                            "chat_id": chat_id,
                            "full_response": analysis_content
                        }
                        yield f"data: {json.dumps(completion_data)}\n\n"
                        
                        # Add analysis to chat history
                        assistant_message = {
                            "role": "assistant",
                            "content": analysis_content,
                            "timestamp": datetime.now().isoformat()
                        }
                        chat_sessions[chat_id].append(assistant_message)
                        return
                    except Exception as e:
                        print(f"Error analyzing UI results: {e}")
                
                # Send completion signal with tool call results if not a handled tool call
                completion_data = {
                    "content": "",
                    "is_complete": True,
                    "chat_id": chat_id,
                    "full_response": tool_calls_str
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
                # Add tool call results to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": tool_calls_str,
                    "timestamp": datetime.now().isoformat()
                }
                chat_sessions[chat_id].append(assistant_message)

            else:
                # Send completion signal with natural response
                completion_data = {
                    "content": "",
                    "is_complete": True,
                    "chat_id": chat_id,
                    "full_response": natural_response
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
                # Add natural response to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": natural_response,
                    "timestamp": datetime.now().isoformat()
                }
                chat_sessions[chat_id].append(assistant_message)
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "is_complete": True,
                "chat_id": chat_id
            }
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
    
    del chat_sessions[chat_id]
    return {"message": f"Chat {chat_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
