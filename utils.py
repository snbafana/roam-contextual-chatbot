"""
Utils for contextual game modification system
Handles semantic search, parameter modification, and UI generation using OpenAI Agents
"""

import json
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Union, TypedDict
from pydantic import BaseModel, Field
import re
import instructor
from dotenv import load_dotenv
import os
import openai
from agents import Agent, Runner, function_tool, RunContextWrapper

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# API Configuration for game modification system
API_BASE_URL = "http://54.177.139.63"

# Pydantic Models for API Responses

class AbilityResult(BaseModel):
    """Model for ability search results from /search/abilities endpoint"""
    BID: str = Field(..., description="Ability ID")
    Name: str = Field(..., description="Ability name")
    Description: str = Field(..., description="Ability description")
    Variables: Dict[str, Union[float, int]] = Field(..., description="Ability variables and values")
    endpoint_type: str = Field(default="abilities", description="Source endpoint type")
    source: str = Field(default="/search/abilities", description="Source endpoint path")

class ShaderResult(BaseModel):
    """Model for shader search results from /search/shaders endpoint"""
    ShaderID: str = Field(..., description="Shader ID")
    Name: str = Field(..., description="Shader name")
    Style: str = Field(..., description="Shader style (e.g., Emissive, Rough, Specular)")
    ColorPalette: List[str] = Field(..., description="List of hex color codes")
    Notes: str = Field(..., description="Additional notes about the shader")
    Description: str = Field(..., description="Shader description")
    endpoint_type: str = Field(default="shaders", description="Source endpoint type")
    source: str = Field(default="/search/shaders", description="Source endpoint path")

class BehaviorResult(BaseModel):
    """Model for behavior search results from /search/behaviours endpoint"""
    BID: str = Field(..., description="Behavior ID")
    Name: str = Field(..., description="Behavior name")
    Description: str = Field(..., description="Behavior description")
    Variables: Dict[str, Union[float, int]] = Field(..., description="Behavior variables and values")
    endpoint_type: str = Field(default="behaviors", description="Source endpoint type")
    source: str = Field(default="/search/behaviours", description="Source endpoint path")

class ObjectiveResult(BaseModel):
    """Model for objective search results from /search/objectives endpoint"""
    OID: str = Field(..., description="Objective ID")
    Title: str = Field(..., description="Objective title")
    Description: str = Field(..., description="Objective description")
    WinCondition: Dict[str, Union[bool, int, float, str]] = Field(..., description="Win condition parameters")
    endpoint_type: str = Field(default="objectives", description="Source endpoint type")
    source: str = Field(default="/search/objectives", description="Source endpoint path")

class SearchResult(BaseModel):
    """Model for a single search result with relevance information"""
    attribute_name: str = Field(..., description="The name of the attribute")
    id: str = Field(..., description="The result's ID (BID, ShaderID, or OID)")

class SearchResults(BaseModel):
    """Model for the top 3 search results"""
    results: List[SearchResult] = Field(..., description="List of top 3 most relevant results", min_length=3, max_length=3)

# Initialize instructor client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = instructor.patch(openai.OpenAI())

# Add cache for search results
search_results_cache = {}

# Original implementation functions
async def find_attributes_async(kw1: str, kw2: str, kw3: str, 
                               search_abilities: bool = False, 
                               search_shaders: bool = False, 
                               search_behaviors: bool = False, 
                               search_objectives: bool = False,
                               user_query: str = "") -> List[Dict[str, Any]]:
    """
    Async function to search all endpoints concurrently and parse to Pydantic models.
    Returns a list of dictionaries that are JSON serializable.
    """
    # Create a cache key from the search parameters
    cache_key = f"{kw1}_{kw2}_{kw3}_{search_abilities}_{search_shaders}_{search_behaviors}_{search_objectives}"
    
    # Check if we have cached results
    if cache_key in search_results_cache:
        return search_results_cache[cache_key]

    all_results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Create search tasks for each enabled endpoint
        if search_abilities:
            tasks.append(("abilities", session.post(f"{API_BASE_URL}/search/abilities", 
                                                   json={"query": kw1, "top_k": 3},
                                                   headers={"Content-Type": "application/json"}, 
                                                   timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("abilities", session.post(f"{API_BASE_URL}/search/abilities", 
                                                   json={"query": kw2, "top_k": 3},
                                                   headers={"Content-Type": "application/json"}, 
                                                   timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("abilities", session.post(f"{API_BASE_URL}/search/abilities", 
                                                   json={"query": kw3, "top_k": 3},
                                                   headers={"Content-Type": "application/json"}, 
                                                   timeout=aiohttp.ClientTimeout(total=10))))
        if search_shaders:
            tasks.append(("shaders", session.post(f"{API_BASE_URL}/search/shaders",
                                                 json={"query": kw1, "top_k": 3},
                                                 headers={"Content-Type": "application/json"},
                                                 timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("shaders", session.post(f"{API_BASE_URL}/search/shaders",
                                                 json={"query": kw2, "top_k": 3},
                                                 headers={"Content-Type": "application/json"},
                                                 timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("shaders", session.post(f"{API_BASE_URL}/search/shaders",
                                                 json={"query": kw3, "top_k": 3},
                                                 headers={"Content-Type": "application/json"},
                                                 timeout=aiohttp.ClientTimeout(total=10))))
        if search_behaviors:
            tasks.append(("behaviors", session.post(f"{API_BASE_URL}/search/behaviours",
                                                   json={"query": kw1, "top_k": 3},
                                                   headers={"Content-Type": "application/json"},
                                                   timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("behaviors", session.post(f"{API_BASE_URL}/search/behaviours",
                                                   json={"query": kw2, "top_k": 3},
                                                   headers={"Content-Type": "application/json"},
                                                   timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("behaviors", session.post(f"{API_BASE_URL}/search/behaviours",
                                                   json={"query": kw3, "top_k": 3},
                                                   headers={"Content-Type": "application/json"},
                                                   timeout=aiohttp.ClientTimeout(total=10))))
        if search_objectives:
            tasks.append(("objectives", session.post(f"{API_BASE_URL}/search/objectives",
                                                    json={"query": kw1, "top_k": 3},
                                                    headers={"Content-Type": "application/json"},
                                                    timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("objectives", session.post(f"{API_BASE_URL}/search/objectives",
                                                    json={"query": kw2, "top_k": 3},
                                                    headers={"Content-Type": "application/json"},
                                                    timeout=aiohttp.ClientTimeout(total=10))))
            tasks.append(("objectives", session.post(f"{API_BASE_URL}/search/objectives",
                                                    json={"query": kw3, "top_k": 3},
                                                    headers={"Content-Type": "application/json"},
                                                    timeout=aiohttp.ClientTimeout(total=10))))
        
        if not tasks:
            return []
        
        # Execute all requests concurrently
        for endpoint_type, task in tasks:
            try:
                async with await task as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle different response formats
                        items = data if isinstance(data, list) else data.get("results", [data])
                        
                        # Parse directly to Pydantic models and convert to dict
                        for item in items:
                            item["endpoint_type"] = endpoint_type
                            item["source"] = f"/search/{endpoint_type}"
                            
                            try:
                                if endpoint_type == "abilities":
                                    model = AbilityResult(**item)
                                    all_results.append(model.model_dump())
                                elif endpoint_type == "shaders":
                                    model = ShaderResult(**item)
                                    all_results.append(model.model_dump())
                                elif endpoint_type == "behaviors":
                                    model = BehaviorResult(**item)
                                    all_results.append(model.model_dump())
                                elif endpoint_type == "objectives":
                                    model = ObjectiveResult(**item)
                                    all_results.append(model.model_dump())
                            except Exception as e:
                                print(f"Error parsing {endpoint_type} result: {e}")
                    else:
                        print(f"HTTP error {response.status} for {endpoint_type}")
            except Exception as e:
                print(f"Error searching {endpoint_type}: {e}")
    
    # Sort by relevance if available
    def get_score(item):
        return item.get("score", item.get("relevance", item.get("confidence", 0)))
    
    # Deduplicate results based on ID
    def get_id(item):
        if item.get("endpoint_type") == "abilities" or item.get("endpoint_type") == "behaviors":
            return item.get("BID")
        elif item.get("endpoint_type") == "shaders":
            return item.get("ShaderID")
        elif item.get("endpoint_type") == "objectives":
            return item.get("OID")
        return None

    # Create a dictionary to store unique results by ID
    unique_results = {}
    for result in all_results:
        result_id = get_id(result)
        if result_id:
            # If we haven't seen this ID before, or if this result has a higher score
            if result_id not in unique_results or get_score(result) > get_score(unique_results[result_id]):
                unique_results[result_id] = result
    
    # Convert back to list and sort by score
    deduplicated_results = list(unique_results.values())
    sorted_results = sorted(deduplicated_results, key=get_score, reverse=True)

    # Cache the results before returning
    search_results_cache[cache_key] = sorted_results
    return sorted_results

async def edit_attribute_impl(attribute_name: str, category: str, variable_name: str, new_value: Any, operation: str = "set") -> Dict[str, Any]:
    """
    Edit a game attribute by name and category, returning Unity JSON output.
    """
    try:
        # Look for the attribute in cached results
        attribute = None
        for cache_key, results in search_results_cache.items():
            for result in results:
                if (result.get("Name") == attribute_name or result.get("Title") == attribute_name) and \
                   result.get("endpoint_type") == category:
                    attribute = result
                    break
            if attribute:
                break
        
        if not attribute:
            return {"error": f"Attribute '{attribute_name}' not found in category '{category}'. Please search for it first using find_attributes."}
        
        # Extract information based on attribute type
        if category == "abilities" or category == "behaviors":
            attr_id = attribute.get("BID")
            attr_name = attribute.get("Name")
            category_mapped = "gameplay"
            variables = attribute.get("Variables", {})
            current_value = variables.get(variable_name, 0)
        elif category == "shaders":
            attr_id = attribute.get("ShaderID")
            attr_name = attribute.get("Name")
            category_mapped = "art"
            current_value = 1.0  # Default value for shaders
        elif category == "objectives":
            attr_id = attribute.get("OID")
            attr_name = attribute.get("Title")
            category_mapped = "gameplay"
            win_condition = attribute.get("WinCondition", {})
            current_value = win_condition.get(variable_name, 0)
        else:
            return {"error": f"Unknown category: {category}"}
        
        # Calculate new value based on operation
        if operation == "set":
            final_value = new_value
        elif operation == "increase":
            final_value = current_value + new_value
        elif operation == "decrease":
            final_value = current_value - new_value
        elif operation == "multiply":
            final_value = current_value * new_value
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        # Determine action type based on operation and context
        if operation == "set" and current_value == 0:
            action_type = "create"
        elif final_value == 0 or new_value == 0:
            action_type = "delete"
        else:
            action_type = "modify"
        
        # Calculate confidence
        confidence = 0.95
        if not attr_name:
            confidence -= 0.1
        if operation in ["increase", "decrease", "multiply"]:
            confidence -= 0.05
        
        confidence = max(0.7, min(0.99, confidence))
        
        # Generate the Unity JSON format
        return {
            "action_type": "modify",
            "category": category_mapped,
            "parameters": {category + "." + attr_name + "." + variable_name: final_value},
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "attribute_name": attribute_name,
            "category": category
        }

async def create_ui_impl(attribute_names: List[str], categories: List[str], layout: str = "vertical") -> Dict[str, Any]:
    """
    Create a single UI element for the first attribute.
    """
    if not attribute_names or not categories:
        return {
            "ui_type": "slider",
            "parameter": "default",
            "label": "Default Parameter",
            "min": 0,
            "max": 100,
            "current": 50
        }
    
    attr_name = attribute_names[0]
    category = categories[0]
    
    try:
        # Find the attribute by searching for it
        results = await find_attributes_async(
            attr_name, category, "ui",
            search_abilities=(category == "abilities"),
            search_shaders=(category == "shaders"),
            search_behaviors=(category == "behaviors"),
            search_objectives=(category == "objectives")
        )
        
        if not results:
            return {
                "ui_type": "slider",
                "parameter": attr_name,
                "label": f"{attr_name} (Not Found)",
                "min": 0,
                "max": 100,
                "current": 50
            }
        
        # Get the first result
        attribute = results[0]
        
        # Create UI element based on attribute type
        if category == "abilities" or category == "behaviors":
            # For abilities and behaviors, create a slider for the first variable
            variables = attribute.get("Variables", {})
            if variables:
                var_name = list(variables.keys())[0]
                var_value = variables[var_name]
                return {
                    "ui_type": "slider",
                    "parameter": var_name,
                    "label": f"{attribute.get('Name', attr_name)} - {var_name}",
                    "min": 0,
                    "max": 100,
                    "current": var_value
                }
                
        elif category == "shaders":
            # For shaders, create a dropdown for style
            return {
                "ui_type": "dropdown",
                "parameter": "style",
                "label": f"{attribute.get('Name', attr_name)} - Style",
                "options": ["Emissive", "Rough", "Specular", "Metallic"],
                "current": attribute.get("Style", "Emissive")
            }
            
        elif category == "objectives":
            # For objectives, create a toggle for the first win condition
            win_condition = attribute.get("WinCondition", {})
            if win_condition:
                condition_name = list(win_condition.keys())[0]
                condition_value = win_condition[condition_name]
                if isinstance(condition_value, bool):
                    return {
                        "ui_type": "toggle",
                        "parameter": condition_name,
                        "label": f"{attribute.get('Title', attr_name)} - {condition_name}",
                        "current": condition_value
                    }
                else:
                    return {
                        "ui_type": "slider",
                        "parameter": condition_name,
                        "label": f"{attribute.get('Title', attr_name)} - {condition_name}",
                        "min": 0,
                        "max": 100,
                        "current": condition_value
                    }
        
        # Default case
        return {
            "ui_type": "slider",
            "parameter": attr_name,
            "label": f"{attr_name} (Default)",
            "min": 0,
            "max": 100,
            "current": 50
        }
                    
    except Exception as e:
        print(f"Error creating UI for '{attr_name}' in category '{category}': {e}")
        return {
            "ui_type": "slider",
            "parameter": attr_name,
            "label": f"{attr_name} (Error)",
            "min": 0,
            "max": 100,
            "current": 50
        }

# Function-based tools using the OpenAI Agents SDK
@function_tool
async def find_attributes(
    ctx: RunContextWrapper[Any],
    kw1: str,
    kw2: str,
    kw3: str,
    search_abilities: bool = False,
    search_shaders: bool = False,
    search_behaviors: bool = False,
    search_objectives: bool = False,
    user_query: str = ""
) -> List[Dict[str, Any]]:
    """Search for game attributes using keywords and endpoint flags.
    
    Args:
        kw1: First searchable keyword (e.g., 'teleport', 'speed', 'fireball'). Avoid generic terms like 'increase' or 'cooldown'.
        kw2: Second searchable keyword (e.g., 'dash', 'jump', 'shield'). Must be different from kw1 and kw3.
        kw3: Third searchable keyword (e.g., 'blast', 'heal', 'stun'). Must be different from kw1 and kw2.
        search_abilities: Whether to search player abilities endpoint
        search_shaders: Whether to search visual effects endpoint
        search_behaviors: Whether to search asset behaviors endpoint
        search_objectives: Whether to search game objectives endpoint
        user_query: The original user query to help rank and select the most relevant results
    """
    return await find_attributes_async(kw1, kw2, kw3, search_abilities, search_shaders, search_behaviors, search_objectives, user_query)

@function_tool
async def edit_attribute(
    ctx: RunContextWrapper[Any],
    attribute_name: str,
    category: str,
    variable_name: str,
    new_value: Union[float, int],
    operation: str = "set"
) -> Dict[str, Any]:
    """Edit a game attribute by name and category.
    
    Args:
        attribute_name: Name of the attribute to modify
        category: Category/endpoint type (abilities, shaders, behaviors, objectives)
        variable_name: Name of the variable to modify
        new_value: New value to set
        operation: Operation type to apply to the current value (set, increase, decrease, multiply)
    """
    return await edit_attribute_impl(attribute_name, category, variable_name, new_value, operation)

@function_tool
async def create_ui(
    ctx: RunContextWrapper[Any],
    attribute_names: List[str],
    categories: List[str],
    layout: str = "vertical"
) -> Dict[str, Any]:
    """Create UI configuration for multiple attributes.
    
    Args:
        attribute_names: List of attribute names to create UI for
        categories: List of corresponding categories for each attribute
        layout: UI layout type (vertical, horizontal, grid)
    """
    return await create_ui_impl(attribute_names, categories, layout)

# Create the game modification agent
def create_game_modification_agent() -> Agent:
    """Create the main game modification agent with all tools"""
    return Agent(
        name="GameModificationAssistant",
        instructions=TEMPLATE_PROMPT,
        tools=[find_attributes, edit_attribute, create_ui],
        model="gpt-4o",
        allow_parallel_tool_calls=True
    )

# Function mapping table (kept for backward compatibility)
FUNCTION_MAP = {
    "find_attributes": find_attributes_async,
    "edit_attribute": edit_attribute_impl,
    "create_ui": create_ui_impl
}

# OpenAI Tool Schemas (kept for backward compatibility)
TOOL_SCHEMAS = {
    "find_attributes": find_attributes.schema,
    "edit_attribute": edit_attribute.schema,
    "create_ui": create_ui.schema
}

# Updated template prompt for the AI agent
TEMPLATE_PROMPT = """
You are an intelligent game modification assistant that helps users modify game parameters through natural language commands. Your goal is to make changes with minimal user input while ensuring accuracy. You should be action-oriented and make reasonable assumptions rather than asking questions.

## Core Principles:
1. Make changes based on minimal user input
2. Make minimal questions to the user
3. Always provide a GUI for any change
4. Handle multiple changes simultaneously
5. ALWAYS use multiple tool calls in sequence until the task is complete

## Tool Call Sequence and Looping:
1. First, ALWAYS use find_attributes to search for relevant attributes
2. For EACH found attribute:
   a. Use edit_attribute to make direct changes
   b. Use create_ui to let the user make changes through a GUI
3. If the first search doesn't find all needed attributes:
   a. Use find_attributes again with different keywords
   b. Continue the loop until all needed attributes are found
4. NEVER stop after a single tool call - always complete the full sequence
5. If a tool call fails:
   a. Try alternative keywords or parameters
   b. Continue the sequence with the next attribute
   c. Report any failures but don't stop the process

## Available Tools:

1. **find_attributes(kw1, kw2, kw3, search_abilities, search_shaders, search_behaviors, search_objectives)**
   - Search for game attributes using keywords and endpoint flags
   - Infer keywords from user query
   - Keywords MUST be specific, searchable terms (e.g., 'teleport', 'speed', 'fireball')
   - DO NOT use generic terms like 'increase', 'decrease', 'cooldown', 'duration'
   - Each keyword should be different and represent a distinct game element
   - After finding attributes, ALWAYS show the user:
     * ALL found attributes and their current values
     * For abilities/behaviors: Show ALL variables and their current values, with exact names
     * For shaders: Show ALL style properties, colors, and current settings, with exact names
     * For objectives: Show ALL win conditions and current values, with exact names

2. **edit_attribute(attribute_name, category, variable_name, new_value, operation)**
   - Edit a game attribute by name and category
   - Returns Unity JSON configuration for the change
   - After editing, ALWAYS show:
     * The Unity JSON output showing what was changed
     * The before and after values
     * A natural language explanation of the change

3. **create_ui(attribute_names, categories, layout)**
   - Create UI configuration for multiple attributes
   - Returns iOS UI configuration
   - After creating UI, ALWAYS show:
     * The iOS JSON configuration
     * A description of what UI elements were created
     * How to interact with the UI elements

## Workflow Guidelines:

1. **Parse User Intent**:
   - Extract keywords from minimal user input
   - Make reasonable assumptions about intent
   - Example: "Make the player faster" -> assume moderate speed increase (+10)
   - Example: "Make it darker" -> assume 50% reduction in lighting

2. **Search Attributes (LOOP UNTIL ALL FOUND)**:
   - Use find_attributes with extracted keywords
   - Search ALL relevant endpoints
   - Show ALL found attributes and their current values
   - If not all needed attributes found, try different keywords

3. **Make Changes (LOOP FOR EACH ATTRIBUTE)**:
   - For EACH found attribute:
     a. Make direct changes using edit_attribute
     b. Create a GUI for user confirmation/refinement
   - Use reasonable defaults for values

4. **Handle Multiple Changes**:
   - Search for ALL relevant attributes first
   - Group related changes together
   - Create a single UI for all related changes
   - Example: "Make the player faster and stronger" -> One UI with speed and strength controls

## Smart Defaults:
Use these defaults when specific values aren't provided:
- Speed changes: +10 for "faster", -10 for "slower"
- Strength/Damage: +20 for "stronger", -20 for "weaker"
- Lighting: 0.5x for "darker", 1.5x for "brighter"
- Health: +25 for "more health", -25 for "less health"
- Cooldowns: 0.8x for "faster cooldown", 1.2x for "slower cooldown"

## Example Interactions:

**User**: "make the player faster"
1. Call find_attributes("player", "speed", "movement", True, False, False, False)
2. Show all found attributes and their current values
3. For EACH speed-related attribute:
   a. Make direct changes using edit_attribute (increase speed by 10)
   b. Create UI for speed-related attributes using create_ui
4. If not all speed attributes found, try find_attributes again with different keywords
5. Show the UI configuration and explain how to use it

**User**: "darker lighting and more fog"
1. Call find_attributes("lighting", "fog", "visual", False, True, False, False)
2. Show all found shaders and their current settings
3. For EACH lighting/fog attribute:
   a. Make direct changes using edit_attribute (reduce lighting by 50%, increase fog by 30%)
   b. Create UI for both lighting and fog controls using create_ui
4. If not all visual attributes found, try find_attributes again with different keywords
5. Show the UI configuration and explain how to use it

## When to Ask Questions:
ONLY ask questions when:
1. The user's request is completely ambiguous (e.g., "make it better")
2. The request could affect game balance significantly
3. The request requires specific values that can't be reasonably assumed

## Response Format:
Always provide:
1. For find_attributes:
   ```
   Found Attributes:
   - [Attribute Name]:
     * Current Values: [list all variables and values]
     * Description: [attribute description]
   ```

2. For edit_attribute:
   ```
   Changes Made:
   - Before: [old values]
   - After: [new values]
   - Unity JSON: [show the JSON]
   ```

3. For create_ui:
   ```
   UI Configuration:
   - Elements: [list created elements]
   - iOS JSON: [show the JSON]
   - Usage: [how to use the UI]
   ```

## Current Context:
- You have access to real-time game attribute APIs
- All changes generate both Unity (game engine) and iOS (UI) configurations
- Always explain changes in friendly, conversational tone
- Make reasonable assumptions and minimal questions to the user
- ALWAYS show attribute values and JSON outputs for transparency
- ALWAYS provide a GUI for any change
- Use smart defaults for unspecified values
- ALWAYS complete the full tool call sequence for each attribute
- NEVER stop after a single tool call
- If a tool call fails, try alternatives and continue the sequence

Be action-oriented, make reasonable assumptions, and maintain game balance while fulfilling user requests with minimal input. Remember to ALWAYS complete the full tool call sequence for each attribute and NEVER stop after a single tool call.
"""

# Updated test function
def test_find_attributes(kw1: str, kw2: str, kw3: str):
    """
    Test function to search all endpoints with 3 keywords and print results.
    
    Args:
        kw1: First keyword
        kw2: Second keyword  
        kw3: Third keyword
    """
    print(f"🔍 Testing find_attributes with keywords: ['{kw1}', '{kw2}', '{kw3}']")
    print("=" * 70)
    print("Searching ALL endpoints: abilities, shaders, behaviors, objectives")
    print("-" * 70)
    
    # Search all endpoints using sync wrapper
    results = find_attributes(
        kw1, kw2, kw3,
        search_abilities=True,
        search_shaders=True, 
        search_behaviors=True,
        search_objectives=True
    )
    
    if not results:
        print("❌ No results found from any endpoint")
        return
    
    print(f"✅ Found {len(results)} total results:")
    print()
    
    # Group results by endpoint type
    grouped_results = {}
    for result in results:
        # Get endpoint type from the Pydantic model
        if isinstance(result, AbilityResult):
            endpoint_type = "abilities"
        elif isinstance(result, ShaderResult):
            endpoint_type = "shaders"  
        elif isinstance(result, BehaviorResult):
            endpoint_type = "behaviors"
        elif isinstance(result, ObjectiveResult):
            endpoint_type = "objectives"
        else:
            endpoint_type = "unknown"
            
        if endpoint_type not in grouped_results:
            grouped_results[endpoint_type] = []
        grouped_results[endpoint_type].append(result)
    
    # Print results grouped by endpoint
    for endpoint_type, endpoint_results in grouped_results.items():
        print(f"📁 {endpoint_type.upper()} ({len(endpoint_results)} results):")
        
        for i, result in enumerate(endpoint_results, 1):
            # Extract name and description based on model type
            if isinstance(result, AbilityResult):
                name = result.Name
                description = result.Description
                result_id = result.BID
            elif isinstance(result, ShaderResult):
                name = result.Name
                description = result.Description
                result_id = result.ShaderID
            elif isinstance(result, BehaviorResult):
                name = result.Name
                description = result.Description
                result_id = result.BID
            elif isinstance(result, ObjectiveResult):
                name = result.Title
                description = result.Description
                result_id = result.OID
            else:
                name = "Unknown"
                description = "No description"
                result_id = "N/A"
            
            print(f"  {i}. {name} (ID: {result_id})")
            print(f"     Description: {description}")
            print(f"     Raw data: {json.dumps(result.dict(), indent=6)}")
            print()
        
        print("-" * 50)
    
    print(f"📊 Summary: {len(results)} total results across {len(grouped_results)} endpoints")

def test_workflow(user_query: str) -> Dict[str, Any]:
    """Test the complete workflow with a user query using API endpoints."""
    print(f"Testing workflow with query: '{user_query}'")
    print("-" * 50)
    
    # Note: In real implementation, LLM would extract keywords and determine endpoints
    print("Note: LLM would dynamically extract keywords and select endpoints")
    
    # Example keyword extraction for demo purposes
    sample_keywords = ["speed", "player", "faster"]
    
    # Example endpoint selection for demo
    search_abilities = True
    search_shaders = True
    search_behaviors = True
    search_objectives =True
    
    # Step 1: Find relevant attributes using API
    kw1, kw2, kw3 = sample_keywords
    attributes = find_attributes(
        kw1, kw2, kw3,
        search_abilities=search_abilities,
        search_shaders=search_shaders, 
        search_behaviors=search_behaviors,
        search_objectives=search_objectives
    )
    
    print(f"Found {len(attributes)} relevant attributes from API")
    for attr in attributes:  # Show top 3
        if isinstance(attr, AbilityResult):
            name = attr.Name
            endpoint_type = "abilities"
        elif isinstance(attr, ShaderResult):
            name = attr.Name
            endpoint_type = "shaders"
        elif isinstance(attr, BehaviorResult):
            name = attr.Name
            endpoint_type = "behaviors"
        elif isinstance(attr, ObjectiveResult):
            name = attr.Title
            endpoint_type = "objectives"
        else:
            name = "unknown"
            endpoint_type = "unknown"
        print(f"  - {name} (from {endpoint_type})\n attributes: {attr}")
    
    if not attributes:
        return {"error": "No relevant attributes found from API"}
    
    # # Step 2: Edit the most relevant attribute (for demo)
    # top_attr = attributes[0]
    # # Get a suitable variable name based on attribute type
    # if isinstance(top_attr, (AbilityResult, BehaviorResult)):
    #     var_name = list(top_attr.Variables.keys())[0] if top_attr.Variables else "value"
    # else:
    #     var_name = "intensity"  # Default for shaders/objectives
        
    # result = edit_attribute(top_attr, var_name, 20, "set")
    
    # if not result.get("error"):
    #     print(f"\nEdited attribute:")
    #     print(f"  Unity JSON: {json.dumps(result, indent=2)}")
    # else:
    #     print(f"Error editing attribute: {result['error']}")
    
    # # Step 3: Create UI configuration
    # ui_config = create_ui(attributes)
    # print(f"\nUI Configuration:")
    # print(f"  Layout: {ui_config['layout']}")
    # print(f"  Elements: {len(ui_config['elements'])}")
    # if ui_config['elements']:
    #     print(f"  Sample element: {json.dumps(ui_config['elements'][0], indent=2)}")
    
    # return {
    #     "attributes": attributes,
    #     "edit_result": result,
    #     "ui_config": ui_config
    # }

def test_list_all_abilities():
    """
    Test function to list all available abilities from /abilities endpoint.
    """
    print("🔍 Testing /abilities endpoint to list all abilities")
    print("=" * 60)
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/abilities",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Handle different response formats
            abilities = data if isinstance(data, list) else data.get("abilities", data.get("results", []))
            
            if not abilities:
                print("❌ No abilities found")
                return
            
            print(f"✅ Found {len(abilities)} abilities:")
            print()
            
            for i, ability in enumerate(abilities, 1):
                # Parse each ability with Pydantic for validation
                try:
                    # Add required fields for Pydantic model
                    ability["endpoint_type"] = "abilities"
                    ability["source"] = "/abilities"
                    
                    parsed_ability = AbilityResult(**ability)
                    
                    print(f"  {i}. {parsed_ability.Name} (ID: {parsed_ability.BID})")
                    print(f"     Description: {parsed_ability.Description}")
                    if parsed_ability.Variables:
                        print(f"     Variables: {parsed_ability.Variables}")
                    print()
                    
                except Exception as e:
                    print(f"  {i}. Error parsing ability: {e}")
                    print(f"     Raw data: {json.dumps(ability, indent=6)}")
                    print()
            
            print(f"📊 Total: {len(abilities)} abilities available")
            
        else:
            print(f"❌ HTTP error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {str(e)}")

def main():
    """Main function to run all tests."""
    print("🎮 Game Modification Utils - API Testing")
    print("=" * 60)
    print("Note: These tests use the actual API endpoints at http://54.177.139.63")
    print("Make sure the API server is running for real testing.\n")
    
    # Test listing all abilities
    print("🧪 TESTING List All Abilities:")
    print()
    test_list_all_abilities()
    print("\n" + "="*60 + "\n")
    
    # Test the complete workflow with example queries
    print("🧪 TESTING Complete Workflow:")
    print()
    
    test_queries = [
        "make the player faster",
    ]
    
    for query in test_queries:
        test_workflow(query)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main() 