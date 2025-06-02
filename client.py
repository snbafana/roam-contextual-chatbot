import gradio as gr
import requests
import json
import asyncio
import aiohttp
from typing import List, Tuple, AsyncGenerator

# API Configuration
API_BASE_URL = "http://localhost:8000"

class ChatClient:
    def __init__(self):
        self.current_chat_id = None
        self.chat_history = []
        self.streaming_enabled = True
    
    def create_new_chat(self):
        """Create a new chat session"""
        try:
            response = requests.post(f"{API_BASE_URL}/create_chat")
            if response.status_code == 200:
                data = response.json()
                self.current_chat_id = data["chat_id"]
                self.chat_history = []
                return f"✅ New chat created! Chat ID: {self.current_chat_id}"
            else:
                return f"❌ Error creating chat: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"❌ Connection error: {str(e)}"
    
    def send_message(self, message: str):
        """Send a message to the current chat (non-streaming)"""
        if not self.current_chat_id:
            return self.chat_history, "❌ Please create a new chat first!"
        
        if not message.strip():
            return self.chat_history, "❌ Please enter a message!"
        
        try:
            # Send message to API
            payload = {"message": message}
            response = requests.post(
                f"{API_BASE_URL}/chat/{self.current_chat_id}/message",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data["response"]
                
                # Update chat history
                self.chat_history.append((message, bot_response))
                
                return self.chat_history, ""
            else:
                error_msg = f"❌ API Error: {response.text}"
                return self.chat_history, error_msg
                
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Connection error: {str(e)}"
            return self.chat_history, error_msg
    
    async def send_message_stream(self, message: str):
        """Send a message to the current chat with streaming response"""
        if not self.current_chat_id:
            yield self.chat_history, "❌ Please create a new chat first!"
            return
        
        if not message.strip():
            yield self.chat_history, "❌ Please enter a message!"
            return
        
        # Add user message to history immediately
        self.chat_history.append((message, ""))
        current_response = ""
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"message": message}
                
                async with session.post(
                    f"{API_BASE_URL}/chat/{self.current_chat_id}/message/stream",
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_msg = f"❌ API Error: {await response.text()}"
                        # Update the last message with error
                        self.chat_history[-1] = (message, error_msg)
                        yield self.chat_history, error_msg
                        return
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                
                                if data.get('error'):
                                    error_msg = f"❌ Streaming Error: {data['error']}"
                                    self.chat_history[-1] = (message, error_msg)
                                    yield self.chat_history, error_msg
                                    return
                                
                                if not data.get('is_complete', False):
                                    # Streaming content
                                    current_response += data.get('content', '')
                                    # Update the last message in history
                                    self.chat_history[-1] = (message, current_response)
                                    yield self.chat_history, ""
                                else:
                                    # Stream completed
                                    if data.get('full_response'):
                                        current_response = data['full_response']
                                        self.chat_history[-1] = (message, current_response)
                                    yield self.chat_history, f"✅ Connected to chat: {self.current_chat_id}"
                                    return
                                    
                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON
                                
        except Exception as e:
            error_msg = f"❌ Connection error: {str(e)}"
            self.chat_history[-1] = (message, error_msg)
            yield self.chat_history, error_msg
    
    def get_chat_list(self):
        """Get list of all active chats"""
        try:
            response = requests.get(f"{API_BASE_URL}/chats")
            if response.status_code == 200:
                data = response.json()
                chat_list = data.get("active_chats", [])
                total_chats = data.get("total_chats", 0)
                
                if chat_list:
                    formatted_list = "\n".join([f"• {chat_id}" for chat_id in chat_list])
                    return f"📋 Active Chats ({total_chats}):\n{formatted_list}"
                else:
                    return "📋 No active chats found."
            else:
                return f"❌ Error getting chat list: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"❌ Connection error: {str(e)}"
    
    def delete_current_chat(self):
        """Delete the current chat session"""
        if not self.current_chat_id:
            return "❌ No active chat to delete!"
        
        try:
            response = requests.delete(f"{API_BASE_URL}/chat/{self.current_chat_id}")
            if response.status_code == 200:
                deleted_chat_id = self.current_chat_id
                self.current_chat_id = None
                self.chat_history = []
                return f"✅ Chat {deleted_chat_id} deleted successfully!"
            else:
                return f"❌ Error deleting chat: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"❌ Connection error: {str(e)}"

# Initialize chat client
chat_client = ChatClient()

# Gradio Interface
def create_chat():
    result = chat_client.create_new_chat()
    return [], result  # Clear chat history and show result

def chat_fn(message, history):
    updated_history, error = chat_client.send_message(message)
    if error:
        # If there's an error, return current history and the error message
        return updated_history, error
    return updated_history, ""

def get_chats():
    return chat_client.get_chat_list()

def delete_chat():
    result = chat_client.delete_current_chat()
    return [], result  # Clear chat history and show result

# Custom CSS for better styling
css = """
.gradio-container {font-family: 'Helvetica Neue', Arial, sans-serif;}
.chat-message {padding: 10px; margin: 5px 0; border-radius: 10px;}
.user-message {background-color: #007bff; color: white; text-align: right;}
.bot-message {background-color: #f8f9fa; color: #333; text-align: left;}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="Contextual Chatbot Client") as demo:
    gr.Markdown("# 🤖 Contextual Chatbot Client")
    gr.Markdown("Connect to your FastAPI backend and chat with GPT-4o-mini!")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat with GPT-4o-mini",
                height=400,
                show_label=True
            )
            
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )
            
            with gr.Row():
                send_btn = gr.Button("Send 📤", variant="primary")
                clear_btn = gr.Button("Clear Chat 🗑️")
                streaming_toggle = gr.Checkbox(
                    label="Enable Streaming 🌊",
                    value=True,
                    info="Stream responses in real-time"
                )
        
        with gr.Column(scale=1):
            # Control panel
            gr.Markdown("### 🎛️ Chat Controls")
            
            create_btn = gr.Button("Create New Chat 🆕", variant="secondary")
            delete_btn = gr.Button("Delete Current Chat ❌", variant="stop")
            
            gr.Markdown("### 📊 Chat Info")
            chat_status = gr.Textbox(
                label="Status",
                value="No active chat. Create a new chat to start!",
                interactive=False,
                lines=3
            )
            
            list_btn = gr.Button("List All Chats 📋")
            chat_list = gr.Textbox(
                label="Active Chats",
                interactive=False,
                lines=5
            )
    
    # Event handlers
    async def send_message_handler(message, history, use_streaming):
        if not message.strip():
            yield history, "", "❌ Please enter a message!"
            return
        
        chat_client.streaming_enabled = use_streaming
        
        if use_streaming:
            # Use streaming
            async for updated_history, error in chat_client.send_message_stream(message):
                status_msg = error if error else f"✅ Streaming... Chat: {chat_client.current_chat_id}"
                yield updated_history, "", status_msg
        else:
            # Use regular non-streaming
            updated_history, error = chat_client.send_message(message)
            status_msg = error if error else f"✅ Connected to chat: {chat_client.current_chat_id}"
            yield updated_history, "", status_msg
    
    def create_chat_handler():
        result = chat_client.create_new_chat()
        return [], result
    
    def delete_chat_handler():
        result = chat_client.delete_current_chat()
        return [], result
    
    def toggle_streaming(enabled):
        chat_client.streaming_enabled = enabled
        return f"🌊 Streaming {'enabled' if enabled else 'disabled'}"
    
    # Wire up the events
    send_btn.click(
        send_message_handler,
        inputs=[msg, chatbot, streaming_toggle],
        outputs=[chatbot, msg, chat_status]
    )
    
    msg.submit(
        send_message_handler,
        inputs=[msg, chatbot, streaming_toggle],
        outputs=[chatbot, msg, chat_status]
    )
    
    streaming_toggle.change(
        toggle_streaming,
        inputs=[streaming_toggle],
        outputs=[chat_status]
    )
    
    create_btn.click(
        create_chat_handler,
        outputs=[chatbot, chat_status]
    )
    
    delete_btn.click(
        delete_chat_handler,
        outputs=[chatbot, chat_status]
    )
    
    clear_btn.click(
        lambda: ([], "Chat cleared locally (server history preserved)"),
        outputs=[chatbot, chat_status]
    )
    
    list_btn.click(
        get_chats,
        outputs=[chat_list]
    )

if __name__ == "__main__":
    print("🚀 Starting Gradio client...")
    print("📡 Make sure your FastAPI server is running on http://localhost:8000")
    print("🔑 Don't forget to set your OPENAI_API_KEY environment variable!")
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True
    ) 