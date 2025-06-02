"""
Utils for contextual game modification system
Handles semantic search, parameter modification, and UI generation
"""

import json
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import re

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
    """Union model that can represent any search result type"""
    result_type: str = Field(..., description="Type of result: ability, shader, behavior, or objective")
    data: Union[AbilityResult, ShaderResult, BehaviorResult, ObjectiveResult] = Field(..., description="The actual result data")

# Helper functions to parse API responses into Pydantic models
def find_attributes(kw1: str, kw2: str, kw3: str, 
                   search_abilities: bool = False, 
                   search_shaders: bool = False, 
                   search_behaviors: bool = False, 
                   search_objectives: bool = False) -> List[Union[AbilityResult, ShaderResult, BehaviorResult, ObjectiveResult]]:
    """
    Sync wrapper to find relevant game attributes using semantic search.
    
    Args:
        kw1: First keyword representing user intent
        kw2: Second keyword representing user intent  
        kw3: Third keyword representing user intent
        search_abilities: Whether to search player abilities endpoint
        search_shaders: Whether to search visual effects endpoint
        search_behaviors: Whether to search asset behaviors endpoint
        search_objectives: Whether to search game objectives endpoint
    
    Returns:
        List of relevant Pydantic model instances from API responses
    """
    return asyncio.run(find_attributes_async(kw1, kw2, kw3, search_abilities, search_shaders, search_behaviors, search_objectives))

async def find_attributes_async(kw1: str, kw2: str, kw3: str, 
                               search_abilities: bool = False, 
                               search_shaders: bool = False, 
                               search_behaviors: bool = False, 
                               search_objectives: bool = False) -> List[Union[AbilityResult, ShaderResult, BehaviorResult, ObjectiveResult]]:
    """
    Async function to search all endpoints concurrently and parse to Pydantic models.
    """
    search_payload = {
        "keywords": [kw1, kw2, kw3],
        "query": f"{kw1} {kw2} {kw3}"
    }
    
    all_results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Create search tasks for each enabled endpoint
        if search_abilities:
            tasks.append(("abilities", session.post(f"{API_BASE_URL}/search/abilities", json=search_payload, 
                                                   headers={"Content-Type": "application/json"}, 
                                                   timeout=aiohttp.ClientTimeout(total=10))))
        if search_shaders:
            tasks.append(("shaders", session.post(f"{API_BASE_URL}/search/shaders", json=search_payload,
                                                 headers={"Content-Type": "application/json"},
                                                 timeout=aiohttp.ClientTimeout(total=10))))
        if search_behaviors:
            tasks.append(("behaviors", session.post(f"{API_BASE_URL}/search/behaviours", json=search_payload,
                                                   headers={"Content-Type": "application/json"},
                                                   timeout=aiohttp.ClientTimeout(total=10))))
        if search_objectives:
            tasks.append(("objectives", session.post(f"{API_BASE_URL}/search/objectives", json=search_payload,
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
                        
                        # Parse directly to Pydantic models
                        for item in items:
                            item["endpoint_type"] = endpoint_type
                            item["source"] = f"/search/{endpoint_type}"
                            
                            try:
                                if endpoint_type == "abilities":
                                    all_results.append(AbilityResult(**item))
                                elif endpoint_type == "shaders":
                                    all_results.append(ShaderResult(**item))
                                elif endpoint_type == "behaviors":
                                    all_results.append(BehaviorResult(**item))
                                elif endpoint_type == "objectives":
                                    all_results.append(ObjectiveResult(**item))
                            except Exception as e:
                                print(f"Error parsing {endpoint_type} result: {e}")
                    else:
                        print(f"HTTP error {response.status} for {endpoint_type}")
            except Exception as e:
                print(f"Error searching {endpoint_type}: {e}")
    
    # Sort by relevance if available
    def get_score(item):
        item_dict = item.dict()
        return item_dict.get("score", item_dict.get("relevance", item_dict.get("confidence", 0)))
    
    return sorted(all_results, key=get_score, reverse=True)

def edit_attribute(attribute_name: str, category: str, variable_name: str, new_value: Any, operation: str = "set") -> Dict[str, Any]:
    """
    Edit a game attribute by name and category, returning Unity JSON output.
    
    Args:
        attribute_name: Name of the attribute to modify
        category: Category/endpoint type (abilities, shaders, behaviors, objectives)
        variable_name: Name of the variable to modify (for abilities/behaviors) or attribute to modify
        new_value: New value to set
        operation: Operation type ("set", "increase", "decrease", "multiply")
    
    Returns:
        Dictionary with Unity JSON configuration
    """
    try:
        # Find the attribute by searching for it
        results = find_attributes(
            attribute_name, category, "modify",
            search_abilities=(category == "abilities"),
            search_shaders=(category == "shaders"),
            search_behaviors=(category == "behaviors"), 
            search_objectives=(category == "objectives")
        )
        
        # Find matching attribute by name
        attribute = None
        for result in results:
            if isinstance(result, AbilityResult) and result.Name == attribute_name:
                attribute = result
                break
            elif isinstance(result, ShaderResult) and result.Name == attribute_name:
                attribute = result
                break
            elif isinstance(result, BehaviorResult) and result.Name == attribute_name:
                attribute = result
                break
            elif isinstance(result, ObjectiveResult) and result.Title == attribute_name:
                attribute = result
                break
        
        if not attribute:
            return {"error": f"Attribute '{attribute_name}' not found in category '{category}'"}
        
        # Extract information based on attribute type
        if isinstance(attribute, AbilityResult):
            attr_id = attribute.BID
            attr_name = attribute.Name
            category_mapped = "gameplay"
            variables = attribute.Variables
            current_value = variables.get(variable_name, 0)
        elif isinstance(attribute, BehaviorResult):
            attr_id = attribute.BID  
            attr_name = attribute.Name
            category_mapped = "asset"
            variables = attribute.Variables
            current_value = variables.get(variable_name, 0)
        elif isinstance(attribute, ShaderResult):
            attr_id = attribute.ShaderID
            attr_name = attribute.Name
            category_mapped = "art"
            current_value = 1.0  # Default value for shaders
        elif isinstance(attribute, ObjectiveResult):
            attr_id = attribute.OID
            attr_name = attribute.Title
            category_mapped = "gameplay"
            win_condition = attribute.WinCondition
            current_value = win_condition.get(variable_name, 0)
        else:
            return {"error": f"Unknown attribute type: {type(attribute)}"}
        
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
            "action_type": action_type,
            "category": category_mapped,
            "parameters": {variable_name: final_value},
            "confidence": round(confidence, 2),
            "attribute_id": attr_id,
            "attribute_name": attr_name
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "attribute_name": attribute_name,
            "category": category
        }

def create_ui(attribute_names: List[str], categories: List[str], layout: str = "vertical") -> Dict[str, Any]:
    """
    Create UI configuration for multiple attributes by name and category.
    
    Args:
        attribute_names: List of attribute names to create UI for
        categories: List of corresponding categories for each attribute
        layout: UI layout type ("vertical", "horizontal", "grid")
    
    Returns:
        iOS UI configuration
    """
    ui_elements = []
    
    for attr_name, category in zip(attribute_names, categories):
        try:
            # Find the attribute by searching for it
            results = find_attributes(
                attr_name, category, "ui",
                search_abilities=(category == "abilities"),
                search_shaders=(category == "shaders"),
                search_behaviors=(category == "behaviors"),
                search_objectives=(category == "objectives")
            )
            
            # Find matching attribute by name
            attribute = None
            for result in results:
                if isinstance(result, AbilityResult) and result.Name == attr_name:
                    attribute = result
                    break
                elif isinstance(result, ShaderResult) and result.Name == attr_name:
                    attribute = result
                    break
                elif isinstance(result, BehaviorResult) and result.Name == attr_name:
                    attribute = result
                    break
                elif isinstance(result, ObjectiveResult) and result.Title == attr_name:
                    attribute = result
                    break
            
            if not attribute:
                print(f"Warning: Attribute '{attr_name}' not found in category '{category}'")
                continue
            
            # Create UI elements based on attribute type
            if isinstance(attribute, (AbilityResult, BehaviorResult)):
                # For abilities and behaviors, create sliders for each variable
                variables = attribute.Variables
                for var_name, var_value in variables.items():
                    ui_element = {
                        "ui_type": "slider",
                        "parameter": var_name,
                        "label": f"{attribute.Name} - {var_name}",
                        "min": 0,
                        "max": 100 if var_name != "cooldown" else 30,
                        "current": var_value
                    }
                    ui_elements.append(ui_element)
                    
            elif isinstance(attribute, ShaderResult):
                # For shaders, create color pickers and style dropdown
                ui_elements.append({
                    "ui_type": "dropdown",
                    "parameter": "style",
                    "label": f"{attribute.Name} - Style",
                    "options": ["Emissive", "Rough", "Specular", "Metallic"],
                    "current": attribute.Style
                })
                
                # Add color palette selector
                for i, color in enumerate(attribute.ColorPalette):
                    ui_elements.append({
                        "ui_type": "color_picker", 
                        "parameter": f"color_{i}",
                        "label": f"{attribute.Name} - Color {i+1}",
                        "current": color
                    })
                    
            elif isinstance(attribute, ObjectiveResult):
                # For objectives, create controls for win conditions
                win_condition = attribute.WinCondition
                for condition_name, condition_value in win_condition.items():
                    if isinstance(condition_value, bool):
                        ui_type = "toggle"
                        ui_element = {
                            "ui_type": ui_type,
                            "parameter": condition_name,
                            "label": f"{attribute.Title} - {condition_name}",
                            "current": condition_value
                        }
                    else:
                        ui_type = "slider"
                        ui_element = {
                            "ui_type": ui_type,
                            "parameter": condition_name,
                            "label": f"{attribute.Title} - {condition_name}",
                            "min": 0,
                            "max": 100,
                            "current": condition_value
                        }
                    ui_elements.append(ui_element)
                    
        except Exception as e:
            print(f"Error processing attribute '{attr_name}' in category '{category}': {e}")
    
    return {
        "layout": layout,
        "elements": ui_elements,
        "total_elements": len(ui_elements),
        "suggested_layout": "grid" if len(ui_elements) > 4 else "vertical"
    }

# Template prompt for the AI agent
TEMPLATE_PROMPT = """
You are an intelligent game modification assistant that helps users modify game parameters through natural language commands.

## Your Workflow:

1. **Parse User Intent**: Extract 3 distinct keywords and determine which endpoints to search
2. **Find Relevant Attributes**: Use find_attributes() with extracted keywords and endpoint flags
3. **Apply Modifications**: Use edit_attribute() to make precise changes
4. **Generate UI**: Use create_ui() to create interface configurations
5. **Provide Context**: Explain changes in natural language

## Intent Classification Guidelines:

### Keyword Extraction:
Extract 3 distinct keywords that capture the user's intent:
- **Primary subject** (player, weapon, environment, character)
- **Action/attribute** (speed, damage, lighting, jump, fire_rate)  
- **Modifier** (faster, stronger, darker, increase, decrease)

### Endpoint Selection:
Choose which endpoints to search based on user intent:

**search_abilities = True** when user mentions:
- Player actions: jump, run, attack, defend, abilities, skills
- Character attributes: health, speed, strength, stamina
- Player mechanics: movement, combat, interaction

**search_shaders = True** when user mentions:
- Visual effects: lighting, shadows, colors, brightness, darkness
- Graphics: shaders, materials, textures, visual style
- Environmental visuals: fog, bloom, contrast, saturation

**search_behaviors = True** when user mentions:
- Asset behaviors: AI, physics, collisions, triggers
- Object interactions: pickup, destruction, animation
- Game mechanics: spawning, pathfinding, state changes

**search_objectives = True** when user mentions:
- Game goals: missions, quests, targets, achievements
- Win conditions: score, time limits, completion criteria
- Progression: levels, unlocks, checkpoints

### Smart Defaults:
- Speed increases: +5-10 units for moderate, +15-20 for significant
- Multipliers: 2x for "double", 1.5x for "boost", 0.5x for "half"
- Health/Damage: +10-25 for small increases, +50-100 for major

### Clarifying Questions:
If user intent is unclear, ask questions like:
- "Are you looking to modify player abilities, visual effects, behaviors, or game objectives?"
- "Do you want to change character attributes, environmental visuals, or game mechanics?"
- "Are you referring to player actions, visual appearance, or game rules?"

### Response Format:
Always provide:
1. Confirmation of what was changed
2. Unity JSON for game engine
3. iOS JSON for UI controls  
4. Natural language explanation

## Example Interactions:

**User**: "make the player faster"
- Keywords: ["player", "speed", "faster"] 
- Endpoints: search_abilities=True
- Response: Enhanced player movement speed

**User**: "darker lighting"  
- Keywords: ["lighting", "dark", "visual"]
- Endpoints: search_shaders=True
- Response: Reduced lighting intensity

**User**: "improve enemy AI"
- Keywords: ["enemy", "ai", "behavior"] 
- Endpoints: search_behaviors=True
- Response: Enhanced AI decision making

## Current Context:
- You have access to real-time game attribute APIs
- All changes generate both Unity (game engine) and iOS (UI) configurations
- Always explain changes in friendly, conversational tone
- If unsure about intent, ask clarifying questions rather than guessing

Be helpful, precise, and maintain game balance while fulfilling user requests.
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