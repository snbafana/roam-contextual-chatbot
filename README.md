# Roam Contextual Chatbot

A FastAPI-based chatbot that integrates with OpenAI's GPT-4o-mini model to provide contextual responses about game attributes, modifications, and UI configurations.

# FYI:
- I am currently streaming the chat back dynamically and passing the json as a code block in that message. I implemented a version where the json is passed directly, but this does not work with streaming, as one cannot pass the json and stream a message at the same time. 
- The attributes in the Unity editor are not validated / verified, they are just the variables initially found. This will need to be change. For example, I pass the `category + "." + attr_name + "." + variable_name` to the Unity json editor, which may not be the exact attribute. 
- The current search system to find inital attributes can take some time to complete, so I am currently working on optimizations here. This is because, for each input, I am dynamically generating 3 keywords for the search, and then deciding which of ALL the results is the most important. 

## Features

- Real-time chat with GPT-4o-mini model
- Tool-based interactions for:
  - Finding game attributes
  - Editing game parameters
  - Creating UI configurations
- Streaming responses for better user experience
- Chat history management
- Logging system for debugging

## Prerequisites

- python
- OpenAI API key
- FastAPI and Uvicorn for the web server
- uv package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/roam-contextual-chatbot.git
cd roam-contextual-chatbot
```

2. Create and activate a virtual environment using uv:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies from the lock file:
```bash
uv pip sync
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the FastAPI server:
```bash
uv run uvicorn main:app --reload
```

2. The API will be available at `http://localhost:8000`

## Testing with the Client

You can use the included client to test the API interactively. The client provides a simple interface to:
- Create new chat sessions
- Send messages and see responses
- View chat history
- Test streaming responses

### Running the Client

1. In a new terminal, with your virtual environment activated, run:
```bash
uv run python client.py
```

2. The client interface will open in your default web browser at `http://localhost:7860`

### Using the Client

1. **Create a New Chat**
   - Click the "Create New Chat" button
   - A new chat session will be created with a unique ID
   - The chat ID will be displayed at the top of the interface

2. **Send Messages**
   - Type your message in the input box
   - Click "Send" or press Enter
   - The response will appear in the chat window
   - For streaming responses, you'll see the text appear gradually

3. **Example Queries**
   ```
   "Find abilities related to player movement"
   "Modify the speed of the sprint ability"
   "Create a UI for the fireball ability"
   ```

4. **View Chat History**
   - All messages and responses are saved in the chat history
   - You can scroll up to view previous interactions
   - The history persists until you create a new chat or close the client

5. **Debug Information**
   - The client shows tool calls and their results
   - You can see the raw API responses
   - Error messages are displayed if something goes wrong

### Client Features

- Real-time streaming of responses
- Automatic chat session management
- Tool call visualization
- Error handling and display
- Chat history persistence
- Easy testing of all API endpoints

## API Endpoints

### Create a New Chat Session
```http
POST /create_chat
```
Response:
```json
{
    "chat_id": "uuid",
    "message": "Chat created successfully with ID: uuid"
}
```

### Send a Message
```http
POST /chat/{chat_id}/message
```
Request body:
```json
{
    "message": "Your message here"
}
```

### Stream a Message
```http
POST /chat/{chat_id}/message/stream
```
Request body:
```json
{
    "message": "Your message here"
}
```

### Get Chat History
```http
GET /chat/{chat_id}/history
```

### List All Chats
```http
GET /chats
```

### Delete a Chat
```http
DELETE /chat/{chat_id}
```

## Available Tools

### find_attributes
Searches for game attributes using keywords and endpoint flags.
```python
find_attributes(
    kw1: str,  # First keyword
    kw2: str,  # Second keyword
    kw3: str,  # Third keyword
    search_abilities: bool = False,
    search_shaders: bool = False,
    search_behaviors: bool = False,
    search_objectives: bool = False,
    user_query: str = ""
)
```

### edit_attribute
Edits a game attribute by name and category.
```python
edit_attribute(
    attribute_name: str,
    category: str,
    variable_name: str,
    new_value: Any,
    operation: str = "set"
)
```

### create_ui
Creates UI configuration for multiple attributes.
```python
create_ui(
    attribute_names: List[str],
    categories: List[str],
    layout: str = "vertical"
)
```

## Logging

Logs are stored in the `logs` directory with the format:
```
logs/chat_{chat_id}_{timestamp}.log
```

## Error Handling

The application includes comprehensive error handling for:
- Invalid chat sessions
- API errors
- Tool execution errors
- Streaming errors


