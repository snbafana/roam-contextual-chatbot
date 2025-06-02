# Roam Contextual Chatbot

A contextual chatbot application built with FastAPI backend and Gradio frontend, integrated with OpenAI's GPT-4o-mini model.

## Features

### FastAPI Backend (`main.py`)
- **Create Chat Sessions**: Generate unique chat IDs for separate conversations
- **Send Messages**: Send messages to specific chats with GPT-4o-mini integration
- **Chat History**: Maintain conversation context across messages
- **Chat Management**: List, view history, and delete chat sessions
- **RESTful API**: Well-documented endpoints with automatic OpenAPI documentation

### Gradio Client (`client.py`)
- **User-Friendly Interface**: Clean chat interface for testing the API
- **Real-time Communication**: Direct integration with FastAPI backend
- **Chat Controls**: Create, delete, and manage multiple chat sessions
- **Status Monitoring**: Real-time connection and chat status updates

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
Set your OpenAI API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the Applications

#### Start the FastAPI Backend
```bash
python main.py
```
The API will be available at: `http://localhost:8000`

#### Start the Gradio Client (in a separate terminal)
```bash
python client.py
```
The client interface will be available at: `http://localhost:7860`

## API Endpoints

### Core Endpoints
- `POST /create_chat` - Create a new chat session
- `POST /chat/{chat_id}/message` - Send a message to a specific chat
- `GET /chat/{chat_id}/history` - Get full chat history
- `GET /chats` - List all active chat sessions
- `DELETE /chat/{chat_id}` - Delete a specific chat session

### API Documentation
Once the FastAPI server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Usage Guide

### Using the Gradio Client
1. **Start both servers** (FastAPI backend and Gradio client)
2. **Open the Gradio interface** at `http://localhost:7860`
3. **Create a new chat** by clicking "Create New Chat 🆕"
4. **Start chatting** by typing messages and clicking "Send 📤"
5. **Manage chats** using the control panel on the right

### Using the API Directly
#### Create a new chat:
```bash
curl -X POST "http://localhost:8000/create_chat"
```

#### Send a message:
```bash
curl -X POST "http://localhost:8000/chat/{chat_id}/message" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?"}'
```

## Project Structure
```
roam-contextual-chatbot/
├── main.py              # FastAPI backend server
├── client.py            # Gradio frontend client
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Default Ports
- **FastAPI Backend**: `8000`
- **Gradio Client**: `7860`

## Features Explained

### Chat Session Management
- Each chat session has a unique UUID
- Chat history is maintained in memory (in production, consider using a database)
- Multiple concurrent chat sessions are supported

### GPT-4o-mini Integration
- Uses OpenAI's GPT-4o-mini model for responses
- Maintains conversation context through message history
- Configurable temperature (0.7) and max tokens (1000)

### Error Handling
- Comprehensive error handling for API calls
- User-friendly error messages in the Gradio interface
- HTTP status codes for API responses

## Development Notes

### Memory Storage
Currently, chat sessions are stored in memory. For production use, consider:
- Database storage (PostgreSQL, MongoDB)
- Redis for session management
- Persistent storage solutions

### Security Considerations
- API key protection
- Rate limiting
- Input validation and sanitization
- CORS configuration for production

### Scaling
- Add authentication and authorization
- Implement rate limiting
- Add database persistence
- Use async database operations
- Add logging and monitoring

## Troubleshooting

### Common Issues
1. **"Connection error"**: Ensure FastAPI server is running on port 8000
2. **"OpenAI API error"**: Check your API key is set correctly and has sufficient credits
3. **"Chat session not found"**: Create a new chat session first

### Debug Mode
Both applications run in debug mode by default for development. For production:
- Set `debug=False` in Gradio
- Configure proper logging in FastAPI
- Use production WSGI server like Gunicorn

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is open source and available under the MIT License.
