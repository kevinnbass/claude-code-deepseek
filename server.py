from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import argparse
import random

# Define Anthropic API URL
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Check for DEBUG environment variable
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "y", "on")

# Configure logging
logging.basicConfig(
    level=logging.INFO if DEBUG_MODE else logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# Flag to enable model swapping between Anthropic and OpenAI
# Always use OpenAI models
USE_OPENAI_MODELS = True

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Gemini API key rotation setup
GEMINI_API_KEYS = [
    "AIzaSyASaK9hy3DWBVcCeLjWgQ21w3EkFzg8Y3c",
    "AIzaSyBq0MikUONa6Scus-n0kHI6hYZW606gpBw",
    "AIzaSyDxQE5xjJP6Og-FNlVUk8qPx3C5MINasPw",
    "AIzaSyAo1s8re6sofxGew941M7sppeeWqWCl7Nc",
    "AIzaSyDHGlVxicUhTN_6jDbK6iBFCPrMU2tOsao",
    "AIzaSyA4nhaky7RnA42biziWqUn2cTNTFJmaNy0",
    "AIzaSyDg938DBcBgHtvmcc1DFpKi8COtHDn4RzU",
    "AIzaSyAkY-cEWW5av3NGUdbH0eKuFBOwkPZfeAw",
    "AIzaSyDRiwSqFCkCecPo-Kj18kHo2NDMXfuxHo4",
    "AIzaSyAIA13wXUQnwUM-VHS3AlQ0ic8beAQn0F0",
    "AIzaSyCoQFwl2uFe-wgGq7quxNBIO8HtrMhBygM",
    "AIzaSyDaCfbee7IuJBen-X4nbrbHenUCNqO6c_o",
    "AIzaSyDkxDNDbRBTMErnPUKKj-l5ONY8mHi16hg",
    "AIzaSyDPYNTlVItzs42Ql9OM6ehqZ5ZeumOMUNI",
    "AIzaSyAvxRnG_JQvtU6GYSUmKY0iHU6uNTSz-Yk",
    "AIzaSyAUzOhJq0z8L6gZaAQuQVNv7hcK3XlWzPE",
    "AIzaSyC4thWdG5LiUSbF_t8VevmQplYUcT9cV9g",
    "AIzaSyDcmKZ6ONssXP1XsoRGVLrMVfckbGrzX7g",
    "AIzaSyCQg155EkEpzFBErh2_LL4d39mjsAYvAZo",
    "AIzaSyASv3CGdQsLyIwWqiV3M8ec9ALr6mPuGHM",
    "AIzaSyD4aj2p2JOw2gnoUg0rNOVXBzsu-RTpcrA",
    "AIzaSyAZtnfkXrSEb4Ij4Hp2uQrdzV08WQwc9p4",
    "AIzaSyCSY4VBE3PuCtm_8sbJK9PCNOlORgbORCU",
    "AIzaSyD8XARYd3OjwCEhDrJIPEwiUP31tLWdrwE",
    "AIzaSyCDs3wH_Q6ADvMAISUO_KrKpqdD81qXXWY",
    "AIzaSyAKdl6eP9EBq55UHo9e9VbGjBc92IREb1A",
    "AIzaSyDJqG8UeoWaAHpITQGNv9kgaTTmB2HmcO0",
    "AIzaSyAaapsepwXKhSiO5IiNza7skZfq84i9gAc",
    "AIzaSyD7_FqyZJoltFAYkZ0HgI6HcnOX_3usffo",
    "AIzaSyAfpqmbgL7HE9bBspGVVT7TLdEDsJplMgU"
]

# Randomly shuffle the initial order of API keys
random.shuffle(GEMINI_API_KEYS)

# Create a counter for key rotation and a variable for the current key
gemini_key_counter = 0
GEMINI_API_KEY = GEMINI_API_KEYS[gemini_key_counter]

# Function to get the next API key in the rotation
def get_next_gemini_api_key():
    global gemini_key_counter, GEMINI_API_KEYS, GEMINI_API_KEY
    
    # Increment counter and wrap around if we reach the end of the list
    gemini_key_counter += 1
    
    # If we've used all keys in the current cycle, shuffle for next cycle
    if gemini_key_counter >= len(GEMINI_API_KEYS):
        gemini_key_counter = 0
        random.shuffle(GEMINI_API_KEYS)
        logger.info(f"Completed a full cycle of Gemini API keys. Reshuffled keys for next cycle.")
    
    # Get the next key
    GEMINI_API_KEY = GEMINI_API_KEYS[gemini_key_counter]
    
    # Log the rotation (only show first few characters of the key for security)
    key_prefix = GEMINI_API_KEY[:8] + "..."
    logger.info(f"Rotating to Gemini API key #{gemini_key_counter+1}/{len(GEMINI_API_KEYS)}: {key_prefix}")
    
    return GEMINI_API_KEY

# Function to clean tool schemas for Gemini compatibility
def clean_gemini_tools(tools):
    """
    Clean tool schemas to be compatible with Gemini API.
    Removes fields like 'additionalProperties' that Gemini doesn't support.
    
    Gemini has several requirements that differ from OpenAI:
    1. No additionalProperties field in schemas
    2. Object types must have non-empty properties
    3. Only 'enum' and 'date-time' formats are supported
    4. Function declarations and parameters must be strictly formatted
    """
    if not tools:
        return tools
    
    cleaned_tools = []
    removed_fields = 0
    
    # Convert tools for Gemini's format - it expects function_declarations
    # This transformation is done by litellm, but we still need to clean the schemas
    for i, tool in enumerate(tools):
        cleaned_tool = tool.copy() if isinstance(tool, dict) else tool
        
        # Clean function schema if present
        if isinstance(cleaned_tool, dict) and "function" in cleaned_tool:
            tool_name = cleaned_tool["function"].get("name", f"tool_{i}")
            
            # Special handling for problematic tools
            if tool_name == "WebFetchTool":
                logger.info(f"Applying special cleanup for WebFetchTool to fix URL format issues")
                fixed_tool = fix_webfetch_tool(cleaned_tool)
                cleaned_tool = fixed_tool
            elif tool_name == "BatchTool":
                logger.info(f"Applying special cleanup for BatchTool")
                fixed_tool = fix_batch_tool(cleaned_tool)
                cleaned_tool = fixed_tool
            elif tool_name == "NotebookEditCell":
                logger.info(f"Applying special cleanup for NotebookEditCell")
                fixed_tool = fix_notebook_edit_cell_tool(cleaned_tool)
                cleaned_tool = fixed_tool
            
            # Handle parameters schema
            if "parameters" in cleaned_tool["function"]:
                # Log the original schema for debugging
                try:
                    original_params = cleaned_tool["function"]["parameters"]
                    logger.debug(f"Original schema for {tool_name} before cleaning: {json.dumps(original_params)[:300]}...")
                except Exception as e:
                    logger.debug(f"Could not serialize original schema for {tool_name}: {str(e)}")
                
                # Clean the schema recursively to handle nested objects
                cleaned_parameters, fields_removed = deep_clean_schema(
                    cleaned_tool["function"]["parameters"], 
                    return_removed_count=True,
                    tool_name=tool_name
                )
                cleaned_tool["function"]["parameters"] = cleaned_parameters
                removed_fields += fields_removed
                
                # Post-cleaning validation - make sure schema conforms to Gemini's expectations
                validate_and_fix_gemini_schema(cleaned_tool["function"], tool_name)
                
                # Log the cleaned schema for debugging
                try:
                    logger.info(f"Cleaned schema for {tool_name}, removed {fields_removed} fields")
                    if fields_removed > 0:
                        logger.debug(f"First 300 chars of cleaned schema: {json.dumps(cleaned_parameters)[:300]}...")
                except Exception as e:
                    logger.debug(f"Could not serialize cleaned schema for {tool_name}: {str(e)}")
                
        cleaned_tools.append(cleaned_tool)
    
    logger.info(f"Cleaned {len(tools)} tool schemas for Gemini compatibility, removed {removed_fields} unsupported fields")
    return cleaned_tools


def validate_and_fix_gemini_schema(function_def, tool_name):
    """
    Final validation pass for a Gemini function schema.
    Ensures all required elements are present and properly formatted.
    """
    try:
        if "parameters" not in function_def:
            logger.warning(f"Tool {tool_name} missing parameters, adding empty parameters")
            function_def["parameters"] = {"type": "object", "properties": {}}
        
        parameters = function_def["parameters"]
        
        # Ensure parameters has required fields
        if "type" not in parameters:
            parameters["type"] = "object"
            
        if "properties" not in parameters:
            parameters["properties"] = {}
            
        # Validate all object types have properties
        if parameters.get("type") == "object" and (not parameters.get("properties") or len(parameters.get("properties", {})) == 0):
            parameters["properties"] = {"_dummy": {"type": "string", "description": "Placeholder property"}}
            logger.info(f"Added dummy property to empty properties object in {tool_name}")
            
        # Fix properties with empty objects
        for prop_name, prop in parameters.get("properties", {}).items():
            if isinstance(prop, dict):
                if prop.get("type") == "object" and "properties" not in prop:
                    prop["properties"] = {"_dummy": {"type": "string", "description": "Placeholder"}}
                    
                # Fix arrays with object items that have no properties
                if prop.get("type") == "array" and "items" in prop:
                    items = prop["items"]
                    if isinstance(items, dict) and items.get("type") == "object" and "properties" not in items:
                        items["properties"] = {"_dummy": {"type": "string", "description": "Placeholder"}}
                        
        # Ensure description is present - required by Gemini
        if "description" not in function_def:
            function_def["description"] = f"Function for {tool_name}"
            logger.info(f"Added missing description to {tool_name}")
    
    except Exception as e:
        logger.error(f"Error validating schema for {tool_name}: {str(e)}")


def fix_batch_tool(tool):
    """
    Special fixes for BatchTool compatibility with Gemini.
    """
    fixed_tool = tool.copy()
    
    try:
        # Get the parameters schema
        params = fixed_tool["function"]["parameters"]
        
        # Fix the invocations structure
        if "properties" in params and "invocations" in params["properties"]:
            invocations_prop = params["properties"]["invocations"]
            
            # Make sure it has the right structure
            if "items" in invocations_prop:
                items = invocations_prop["items"]
                
                # Remove additionalProperties if present
                if "additionalProperties" in items:
                    del items["additionalProperties"]
                    logger.info(f"Removed additionalProperties from BatchTool items")
                
                # Fix the input object - this is the main issue with BatchTool
                if "properties" in items and "input" in items["properties"]:
                    input_prop = items["properties"]["input"]
                    
                    # Remove any format fields
                    if "format" in input_prop:
                        del input_prop["format"]
                    
                    # If input is an object with no properties
                    if input_prop.get("type") == "object" and "properties" not in input_prop:
                        # Add concrete properties (Gemini requires these)
                        input_prop["properties"] = {
                            "command": {"type": "string", "description": "Command parameter"},
                            "file_path": {"type": "string", "description": "File path parameter"},
                            "prompt": {"type": "string", "description": "Prompt parameter"},
                            "pattern": {"type": "string", "description": "Pattern parameter"},
                            "path": {"type": "string", "description": "Path parameter"}
                        }
                        logger.info(f"Added concrete properties structure to BatchTool input object")
                        
    except Exception as e:
        logger.error(f"Error fixing BatchTool schema: {str(e)}")
    
    return fixed_tool


def fix_notebook_edit_cell_tool(tool):
    """
    Special fixes for NotebookEditCell tool compatibility with Gemini.
    """
    fixed_tool = tool.copy()
    
    try:
        # Get the parameters schema
        params = fixed_tool["function"]["parameters"]
        
        # Remove enum fields which can cause issues with Gemini
        if "properties" in params:
            props = params["properties"]
            
            # Fix the cell_type enum - Gemini has specific format for enums
            if "cell_type" in props and "enum" in props["cell_type"]:
                # Convert enum to string type with description listing allowed values
                enum_values = props["cell_type"]["enum"]
                props["cell_type"] = {
                    "type": "string",
                    "description": f"The type of cell. Allowed values: {', '.join(enum_values)}"
                }
                logger.info(f"Converted cell_type enum to string with description for NotebookEditCell")
            
            # Fix the edit_mode field - Gemini prefers string over enum
            if "edit_mode" in props:
                # Make sure it's just a simple string type
                props["edit_mode"] = {
                    "type": "string",
                    "description": "The type of edit to make (replace, insert, delete). Defaults to replace."
                }
                logger.info(f"Simplified edit_mode field for NotebookEditCell")
                
    except Exception as e:
        logger.error(f"Error fixing NotebookEditCell schema: {str(e)}")
    
    return fixed_tool


def fix_webfetch_tool(tool):
    """
    Special fix for WebFetchTool to address the URL format issue.
    Gemini only supports 'enum' and 'date-time' formats.
    """
    fixed_tool = tool.copy()
    
    try:
        # Get the parameters schema
        params = fixed_tool["function"]["parameters"]
        
        # Fix URL property if it exists
        if ("properties" in params and 
            "url" in params["properties"] and 
            "format" in params["properties"]["url"]):
            
            # Remove the format field completely
            if params["properties"]["url"]["format"] == "uri":
                logger.info("Removing 'uri' format from WebFetchTool url parameter")
                del params["properties"]["url"]["format"]
                
    except Exception as e:
        logger.error(f"Error fixing WebFetchTool schema: {str(e)}")
    
    return fixed_tool

def deep_clean_schema(schema, return_removed_count=False, tool_name=None, path=""):
    """
    Recursively clean JSON schema to remove fields not supported by Gemini.
    
    Args:
        schema: The JSON schema to clean
        return_removed_count: If True, returns a tuple of (cleaned_schema, removed_count)
        tool_name: Name of the tool being processed (for logging)
        path: Current path in the schema (for recursion)
        
    Returns:
        The cleaned schema, or a tuple of (cleaned_schema, removed_count) if return_removed_count is True
    """
    if not isinstance(schema, dict):
        return (schema, 0) if return_removed_count else schema
        
    # Fields to remove from schema that Gemini doesn't support
    fields_to_remove = [
        "additionalProperties", 
        "$schema", 
        "$id", 
        "examples", 
        "format",
        "patterns",
        "contentMediaType",
        "contentEncoding"
    ]
    
    # Count how many fields we remove
    removed_count = 0
    current_path = f"{path}" if not path else f"{path}."
    
    # Create a new dict without forbidden fields
    cleaned_schema = {}
    for k, v in schema.items():
        if k not in fields_to_remove:
            cleaned_schema[k] = v
        else:
            field_path = f"{current_path}{k}"
            if tool_name:
                logger.debug(f"Removed field '{field_path}' from tool '{tool_name}'")
            removed_count += 1
    
    # Handle URIs and other formats that Gemini doesn't support
    # Only 'enum' and 'date-time' formats are supported
    if "format" in schema:
        format_val = schema["format"]
        if format_val not in ["enum", "date-time"]:
            field_path = f"{current_path}format"
            if tool_name:
                logger.info(f"Removed unsupported format '{format_val}' at '{field_path}' from tool '{tool_name}'")
            
            # Remove the format field if still in cleaned schema
            if "format" in cleaned_schema:
                del cleaned_schema["format"]
                removed_count += 1
    
    # Fix empty object properties - Gemini requires non-empty properties for objects
    if cleaned_schema.get("type") == "object" and "properties" not in cleaned_schema:
        cleaned_schema["properties"] = {"_placeholder": {"type": "string", "description": "Required placeholder"}}
        field_path = f"{current_path}properties._placeholder"
        if tool_name:
            logger.info(f"Added required properties at '{field_path}' for tool '{tool_name}'")
        removed_count += 1  # Count adding the placeholder as a change
    
    # Recursively clean nested objects
    for key, value in list(cleaned_schema.items()):
        new_path = f"{current_path}{key}"
        
        if key == "properties" and isinstance(value, dict):
            # Clean each property
            for prop_name, prop_schema in list(value.items()):
                prop_path = f"{new_path}.{prop_name}"
                if return_removed_count:
                    cleaned_prop, prop_removed = deep_clean_schema(
                        prop_schema, 
                        return_removed_count=True,
                        tool_name=tool_name,
                        path=prop_path
                    )
                    cleaned_schema[key][prop_name] = cleaned_prop
                    removed_count += prop_removed
                else:
                    cleaned_schema[key][prop_name] = deep_clean_schema(
                        prop_schema,
                        tool_name=tool_name,
                        path=prop_path
                    )
                    
                # Fix empty object types
                if (isinstance(cleaned_schema[key][prop_name], dict) and 
                    cleaned_schema[key][prop_name].get("type") == "object" and 
                    "properties" not in cleaned_schema[key][prop_name]):
                    
                    cleaned_schema[key][prop_name]["properties"] = {
                        "_placeholder": {"type": "string", "description": "Required placeholder"}
                    }
                    if tool_name:
                        logger.info(f"Added required nested properties at '{prop_path}' for tool '{tool_name}'")
                    removed_count += 1
                    
        elif key == "items" and isinstance(value, dict):
            # Clean array item schema
            items_path = f"{new_path}"
            if return_removed_count:
                cleaned_items, items_removed = deep_clean_schema(
                    value, 
                    return_removed_count=True,
                    tool_name=tool_name,
                    path=items_path
                )
                cleaned_schema[key] = cleaned_items
                removed_count += items_removed
            else:
                cleaned_schema[key] = deep_clean_schema(
                    value,
                    tool_name=tool_name,
                    path=items_path
                )
                
            # Fix empty object types in array items
            if (isinstance(cleaned_schema[key], dict) and 
                cleaned_schema[key].get("type") == "object" and 
                "properties" not in cleaned_schema[key]):
                
                cleaned_schema[key]["properties"] = {
                    "_placeholder": {"type": "string", "description": "Required placeholder"}
                }
                if tool_name:
                    logger.info(f"Added required properties to array items at '{items_path}' for tool '{tool_name}'")
                removed_count += 1
                
        elif isinstance(value, dict):
            # Clean any nested dict
            nested_path = f"{new_path}"
            if return_removed_count:
                cleaned_dict, dict_removed = deep_clean_schema(
                    value, 
                    return_removed_count=True,
                    tool_name=tool_name,
                    path=nested_path
                )
                cleaned_schema[key] = cleaned_dict
                removed_count += dict_removed
            else:
                cleaned_schema[key] = deep_clean_schema(
                    value,
                    tool_name=tool_name,
                    path=nested_path
                )
                
        elif isinstance(value, list):
            # Clean list of objects
            list_path = f"{new_path}"
            if return_removed_count:
                cleaned_list = []
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        cleaned_item, item_removed = deep_clean_schema(
                            item, 
                            return_removed_count=True,
                            tool_name=tool_name,
                            path=f"{list_path}[{i}]"
                        )
                        cleaned_list.append(cleaned_item)
                        removed_count += item_removed
                    else:
                        cleaned_list.append(item)
                cleaned_schema[key] = cleaned_list
            else:
                cleaned_schema[key] = [
                    deep_clean_schema(
                        item, 
                        tool_name=tool_name,
                        path=f"{list_path}[{i}]"
                    ) if isinstance(item, dict) else item
                    for i, item in enumerate(value)
                ]
    
    # Final check - special case for specific tools with problematic fields
    if tool_name == "BatchTool" and path == "parameters.properties.invocations.items.properties.input":
        # Make sure BatchTool's input parameter has required properties
        if "type" in cleaned_schema and cleaned_schema["type"] == "object" and "properties" not in cleaned_schema:
            cleaned_schema["properties"] = {
                "command": {"type": "string", "description": "Command parameter"},
                "file_path": {"type": "string", "description": "File path parameter"}
            }
            logger.info(f"Added specific BatchTool input properties at '{path}'")
            removed_count += 1
    
    # Fix WebFetchTool's URL format
    if tool_name == "WebFetchTool" and "format" in cleaned_schema:
        if cleaned_schema["format"] == "uri":
            del cleaned_schema["format"]
            logger.info(f"Removed 'uri' format from WebFetchTool at path '{path}'")
            removed_count += 1
    
    if return_removed_count:
        return cleaned_schema, removed_count
    return cleaned_schema

# Get model mapping configuration from environment
# deepseek-chat is recommended for all tasks including coding
BIG_MODEL = os.environ.get("BIG_MODEL", "deepseek-chat")
BIG_MODEL_PROVIDER = os.environ.get("BIG_MODEL_PROVIDER", "deepseek")  # Can be "deepseek" or "gemini"
GEMINI_BIG_MODEL = os.environ.get("GEMINI_BIG_MODEL", "gemini-2.5-pro-exp-03-25")  # Gemini model for Sonnet
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-2.0-flash")  # Default to Gemini Flash for Haiku

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Anthropic Proxy Server for Deepseek and Gemini")
parser.add_argument('--always-cot', action='store_true', help='Always add Chain-of-Thought system prompt for Sonnet models')
args, _ = parser.parse_known_args()
ALWAYS_COT = args.always_cot

if ALWAYS_COT:
    logger.warning("🧠 ALWAYS_COT mode activated: Chain-of-Thought will be added to all Sonnet model requests")

# Chain of Thought system prompt for reasoning (used only when Sonnet models are mapped to Deepseek)
COT_SYSTEM_PROMPT = "You are a helpful assistant that uses chain-of-thought reasoning. For complex questions, always break down your reasoning step-by-step before giving an answer."

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = Field(default=None)
    original_model: Optional[str] = None  # Will store the original model name
    
    model_config = {
        "extra": "allow"  # Allow extra fields for forward compatibility
    }
    
    @field_validator('model')
    def validate_model(cls, v, info):
        # Store the original model name
        original_model = v
        
        # Check if we're using Deepseek/Gemini models and need to swap
        if USE_OPENAI_MODELS:
            # Remove anthropic/ prefix if it exists
            if v.startswith('anthropic/'):
                v = v[10:]  # Remove 'anthropic/' prefix
            
            # Swap Haiku with Gemini Flash model
            if 'haiku' in v.lower():
                # Use Gemini model for Haiku requests
                new_model = f"gemini/{SMALL_MODEL}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                v = new_model
            
            # Handle Sonnet models
            elif 'sonnet' in v.lower():
                # Map Sonnet to either Deepseek or Gemini based on BIG_MODEL_PROVIDER setting
                if BIG_MODEL_PROVIDER.lower() == "gemini":
                    new_model = f"gemini/{GEMINI_BIG_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (Gemini provider)")
                else:  # Default to Deepseek
                    new_model = f"deepseek/{BIG_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (Deepseek provider)")
                v = new_model
                
                # Check if thinking is enabled to decide whether to add CoT
                values = info.data
                if isinstance(values, dict):
                    # Check if ALWAYS_COT flag is set
                    if ALWAYS_COT:
                        thinking_enabled = True
                        logger.debug(f"📌 ALWAYS_COT enabled: Adding CoT system prompt for Sonnet model")
                    else:
                        # Check for thinking parameter
                        thinking_enabled = False
                        thinking_data = values.get("thinking")
                        
                        # Handle all possible thinking formats
                        if thinking_data is not None:
                            # Boolean value
                            if isinstance(thinking_data, bool):
                                thinking_enabled = thinking_data
                            # Dict with enabled key
                            elif isinstance(thinking_data, dict) and 'enabled' in thinking_data:
                                thinking_enabled = bool(thinking_data['enabled'])
                            # Empty dict - treat as enabled=True
                            elif isinstance(thinking_data, dict):
                                thinking_enabled = True
                            # Any other value - presence means enabled
                            else:
                                thinking_enabled = True
                    
                    if thinking_enabled:
                        # Add Chain-of-Thought system prompt when thinking is enabled
                        logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} with CoT (thinking enabled)")
                        
                        # If system prompt already exists, prepend CoT to it
                        if values.get("system"):
                            if isinstance(values["system"], str):
                                values["system"] = f"{COT_SYSTEM_PROMPT}\n\n{values['system']}"
                            # If it's a list, add CoT to the beginning
                            elif isinstance(values["system"], list):
                                values["system"].insert(0, {"type": "text", "text": COT_SYSTEM_PROMPT})
                        else:
                            # No system prompt exists, add the CoT system prompt
                            values["system"] = COT_SYSTEM_PROMPT
                    else:
                        # No CoT for normal mode
                        logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (normal mode)")
            
            # Keep the model as is but add deepseek/ prefix if not already present
            elif not v.startswith('deepseek/'):
                new_model = f"deepseek/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                v = new_model
                
            # Store the original model in the values dictionary
            # This will be accessible as request.original_model
            values = info.data
            if isinstance(values, dict):
                values['original_model'] = original_model
                
            return v
        else:
            # Original behavior - ensure anthropic/ prefix
            original_model = v
            if not v.startswith('anthropic/'):
                new_model = f"anthropic/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                
                # Store original model
                values = info.data
                if isinstance(values, dict):
                    values['original_model'] = original_model
                    
                return new_model
            return v

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[Dict[str, Any]] = Field(default=None)
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    model_config = {
        "extra": "allow"  # Allow extra fields for forward compatibility
    }
    
    @field_validator('model')
    def validate_model(cls, v, info):
        # Store the original model name
        original_model = v
        
        # Same validation as MessagesRequest
        if USE_OPENAI_MODELS:
            # Remove anthropic/ prefix if it exists
            if v.startswith('anthropic/'):
                v = v[10:]  
            
            # Swap Haiku with Gemini Flash model
            if 'haiku' in v.lower():
                # Use Gemini model for Haiku requests
                new_model = f"gemini/{SMALL_MODEL}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                v = new_model
            
            # Handle Sonnet models
            elif 'sonnet' in v.lower():
                # Map Sonnet to either Deepseek or Gemini based on BIG_MODEL_PROVIDER setting
                if BIG_MODEL_PROVIDER.lower() == "gemini":
                    new_model = f"gemini/{GEMINI_BIG_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (Gemini provider)")
                else:  # Default to Deepseek
                    new_model = f"deepseek/{BIG_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (Deepseek provider)")
                v = new_model
                
                # Check if thinking is enabled to decide whether to add CoT
                values = info.data
                if isinstance(values, dict):
                    # Check if ALWAYS_COT flag is set
                    if ALWAYS_COT:
                        thinking_enabled = True
                        logger.debug(f"📌 ALWAYS_COT enabled: Adding CoT system prompt for Sonnet model")
                    else:
                        # Check for thinking parameter
                        thinking_enabled = False
                        thinking_data = values.get("thinking")
                        
                        # Handle all possible thinking formats
                        if thinking_data is not None:
                            # Boolean value
                            if isinstance(thinking_data, bool):
                                thinking_enabled = thinking_data
                            # Dict with enabled key
                            elif isinstance(thinking_data, dict) and 'enabled' in thinking_data:
                                thinking_enabled = bool(thinking_data['enabled'])
                            # Empty dict - treat as enabled=True
                            elif isinstance(thinking_data, dict):
                                thinking_enabled = True
                            # Any other value - presence means enabled
                            else:
                                thinking_enabled = True
                    
                    if thinking_enabled:
                        # Add Chain-of-Thought system prompt when thinking is enabled
                        logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} with CoT (thinking enabled)")
                        
                        # If system prompt already exists, prepend CoT to it
                        if values.get("system"):
                            if isinstance(values["system"], str):
                                values["system"] = f"{COT_SYSTEM_PROMPT}\n\n{values['system']}"
                            # If it's a list, add CoT to the beginning
                            elif isinstance(values["system"], list):
                                values["system"].insert(0, {"type": "text", "text": COT_SYSTEM_PROMPT})
                        else:
                            # No system prompt exists, add the CoT system prompt
                            values["system"] = COT_SYSTEM_PROMPT
                    else:
                        # No CoT for normal mode
                        logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (normal mode)")
            
            # Keep the model as is but add deepseek/ prefix if not already present
            elif not v.startswith('deepseek/'):
                new_model = f"deepseek/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                v = new_model
            
            # Store the original model in the values dictionary
            values = info.data
            if isinstance(values, dict):
                values['original_model'] = original_model
                
            return v
        else:
            # Original behavior - ensure anthropic/ prefix
            if not v.startswith('anthropic/'):
                new_model = f"anthropic/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                
                # Store original model
                values = info.data
                if isinstance(values, dict):
                    values['original_model'] = original_model
                    
                return new_model
            return v

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response


# Create a parameter mapper to handle common parameter naming mismatches
def map_tool_parameters(tool_call):
    """Maps and fixes common parameter naming issues in tool calls"""
    try:
        logger.warning(f"TOOL DEBUG: map_tool_parameters input type: {type(tool_call)}")
        
        if not tool_call:
            logger.warning("TOOL DEBUG: tool_call is None or empty")
            return tool_call
            
        # Handle different object types correctly
        function_dict = {}
        name = ""
        arguments = {}
        
        if hasattr(tool_call, 'function'):
            # It's an object with a function attribute
            func = tool_call.function
            if hasattr(func, 'name'):
                name = func.name
            if hasattr(func, 'arguments'):
                arguments = func.arguments
                
            logger.warning(f"TOOL DEBUG: Extracted from object - name: {name}, arguments: {arguments}")
        elif isinstance(tool_call, dict) and 'function' in tool_call:
            # It's a dict with a function key
            function_dict = tool_call['function']
            if isinstance(function_dict, dict):
                name = function_dict.get('name', '')
                arguments = function_dict.get('arguments', {})
            else:
                # Function is an object
                if hasattr(function_dict, 'name'):
                    name = function_dict.name
                if hasattr(function_dict, 'arguments'):
                    arguments = function_dict.arguments
                    
            logger.warning(f"TOOL DEBUG: Extracted from dict - name: {name}, arguments: {arguments}")
        else:
            logger.warning(f"TOOL DEBUG: Unsupported tool call format")
            return tool_call
            
        # Ensure arguments is a dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
                logger.warning(f"TOOL DEBUG: Parsed arguments from string to dict: {arguments}")
            except json.JSONDecodeError as e:
                logger.warning(f"TOOL DEBUG: Failed to parse arguments JSON: {e}")
                
                # Try to apply simple fixes to the JSON
                args_str = arguments
                if not args_str.strip().startswith('{'):
                    args_str = '{' + args_str
                if not args_str.strip().endswith('}'):
                    args_str = args_str + '}'
                    
                # Try again
                try:
                    arguments = json.loads(args_str)
                    logger.warning(f"TOOL DEBUG: Fixed and parsed arguments: {arguments}")
                except:
                    # Give up and return original
                    logger.warning(f"TOOL DEBUG: Could not fix JSON")
                    return tool_call
        
        # Apply parameter mappings to normalize names
        # Dictionary of parameter mappings by tool name
        parameter_mappings = {
            'View': {
                'path': 'file_path',
                'file': 'file_path',
                'filename': 'file_path',
                'filepath': 'file_path',
                'lines': 'limit',
                'start_line': 'offset',
                'from_line': 'offset',
                'line_count': 'limit',
                'max_lines': 'limit'
            },
            'Edit': {
                'path': 'file_path',
                'file': 'file_path',
                'filename': 'file_path',
                'filepath': 'file_path',
                'content': 'new_string',
                'new_content': 'new_string',
                'replacement': 'new_string',
                'text': 'old_string',
                'old_content': 'old_string',
                'target': 'old_string',
                'to_replace': 'old_string'
            },
            'Replace': {
                'path': 'file_path',
                'file': 'file_path',
                'filename': 'file_path',
                'filepath': 'file_path',
                'text': 'content',
                'new_content': 'content',
                'data': 'content'
            },
            'ReadNotebook': {
                'path': 'notebook_path',
                'file': 'notebook_path',
                'filename': 'notebook_path',
                'filepath': 'notebook_path',
                'notebook': 'notebook_path'
            },
            'NotebookEditCell': {
                'path': 'notebook_path',
                'file': 'notebook_path',
                'filename': 'notebook_path',
                'filepath': 'notebook_path',
                'notebook': 'notebook_path',
                'cell': 'cell_number',
                'index': 'cell_number',
                'cell_index': 'cell_number',
                'source': 'new_source',
                'content': 'new_source',
                'cell_content': 'new_source',
                'mode': 'edit_mode',
                'type': 'cell_type'
            },
            'GlobTool': {
                'directory': 'path',
                'dir': 'path',
                'search_path': 'path',
                'glob': 'pattern',
                'search': 'pattern',
                'query': 'pattern'
            },
            'GrepTool': {
                'directory': 'path',
                'dir': 'path',
                'search_path': 'path',
                'search': 'pattern',
                'query': 'pattern',
                'text': 'pattern',
                'regex': 'pattern',
                'file_type': 'include',
                'extension': 'include',
                'file_pattern': 'include'
            },
            'LS': {
                'directory': 'path',
                'dir': 'path',
                'folder': 'path',
                'exclude': 'ignore',
                'excluded': 'ignore',
                'ignore_patterns': 'ignore'
            },
            'WebFetchTool': {
                'website': 'url',
                'link': 'url',
                'uri': 'url',
                'query': 'prompt',
                'question': 'prompt',
                'instruction': 'prompt'
            },
            'dispatch_agent': {
                'task': 'prompt',
                'instruction': 'prompt',
                'query': 'prompt',
                'question': 'prompt'
            },
            'BatchTool': {
                'description': 'description',
                'calls': 'invocations',
                'tools': 'invocations'
            }
        }
        
        # Apply mappings if tool name exists in our mapping dictionary
        if name in parameter_mappings:
            mappings = parameter_mappings[name]
            updated_arguments = {}
            
            # Copy all existing arguments
            for key, value in arguments.items():
                # If this key should be mapped to another name
                if key in mappings:
                    mapped_key = mappings[key]
                    updated_arguments[mapped_key] = value
                    logger.warning(f"PARAMETER MAP: Mapped tool parameter {key} -> {mapped_key} for tool {name}")
                else:
                    updated_arguments[key] = value
                    
            # Special handling for Edit tool if parameters are missing
            if name == 'Edit':
                # Create a blank old_string if we're doing a file creation operation
                if 'file_path' in updated_arguments and 'new_string' in updated_arguments and 'old_string' not in updated_arguments:
                    updated_arguments['old_string'] = ''
                    logger.warning(f"PARAMETER MAP: Added empty old_string parameter for Edit tool with file_path and new_string")
                elif 'path' in arguments and 'content' in arguments and 'old_string' not in updated_arguments:
                    # Check if we need to map more parameters
                    if 'path' in arguments and 'path' not in mappings:
                        updated_arguments['file_path'] = arguments['path']
                    if 'content' in arguments and 'content' not in mappings:
                        updated_arguments['new_string'] = arguments['content']
                    updated_arguments['old_string'] = ''
                    logger.warning(f"PARAMETER MAP: Added empty old_string parameter for Edit tool with path and content")
            
            # Special handling for Replace tool if parameters are missing
            elif name == 'Replace':
                if 'file_path' in updated_arguments and 'content' not in updated_arguments:
                    # Try to find content in other parameters
                    for param in ['text', 'new_content', 'data']:
                        if param in arguments:
                            updated_arguments['content'] = arguments[param]
                            logger.warning(f"PARAMETER MAP: Mapped {param} to content for Replace tool")
                            break
            
            # Special handling for View tool if parameters are missing expected format
            elif name == 'View':
                # Convert string offsets and limits to integers
                for param in ['offset', 'limit']:
                    if param in updated_arguments and isinstance(updated_arguments[param], str):
                        try:
                            updated_arguments[param] = int(updated_arguments[param])
                            logger.warning(f"PARAMETER MAP: Converted {param} from string to integer for View tool")
                        except ValueError:
                            logger.warning(f"PARAMETER MAP: Could not convert {param} value '{updated_arguments[param]}' to integer")
            
            # Special handling for NotebookEditCell tool
            elif name == 'NotebookEditCell':
                # Convert cell_number from string to integer if needed
                if 'cell_number' in updated_arguments and isinstance(updated_arguments['cell_number'], str):
                    try:
                        updated_arguments['cell_number'] = int(updated_arguments['cell_number'])
                        logger.warning(f"PARAMETER MAP: Converted cell_number from string to integer for NotebookEditCell tool")
                    except ValueError:
                        logger.warning(f"PARAMETER MAP: Could not convert cell_number value '{updated_arguments['cell_number']}' to integer")
            
            arguments = updated_arguments
            
        # Construct the fixed tool call
        result = {
            'id': getattr(tool_call, 'id', f"tool_{uuid.uuid4()}") if hasattr(tool_call, 'id') else tool_call.get('id', f"tool_{uuid.uuid4()}"),
            'type': 'function',
            'function': {
                'name': name,
                'arguments': arguments
            }
        }
        
        logger.warning(f"TOOL DEBUG: Mapped tool call: {json.dumps(result, default=str)}")
        return result
        
    except Exception as e:
        logger.error(f"Error in map_tool_parameters: {str(e)}")
        # Return the input unchanged on error
        return tool_call
    
    # Dictionary of parameter mappings by tool name
    parameter_mappings = {
        'View': {
            'path': 'file_path',
            'file': 'file_path',
            'filename': 'file_path',
            'filepath': 'file_path',
            'lines': 'limit',
            'start_line': 'offset',
            'from_line': 'offset',
            'line_count': 'limit',
            'max_lines': 'limit'
        },
        'Edit': {
            'path': 'file_path',
            'file': 'file_path',
            'filename': 'file_path',
            'filepath': 'file_path',
            'content': 'new_string',
            'new_content': 'new_string',
            'replacement': 'new_string',
            'text': 'old_string',
            'old_content': 'old_string',
            'target': 'old_string',
            'to_replace': 'old_string'
        },
        'Replace': {
            'path': 'file_path',
            'file': 'file_path',
            'filename': 'file_path',
            'filepath': 'file_path',
            'text': 'content',
            'new_content': 'content',
            'data': 'content'
        },
        'ReadNotebook': {
            'path': 'notebook_path',
            'file': 'notebook_path',
            'filename': 'notebook_path',
            'filepath': 'notebook_path',
            'notebook': 'notebook_path'
        },
        'NotebookEditCell': {
            'path': 'notebook_path',
            'file': 'notebook_path',
            'filename': 'notebook_path',
            'filepath': 'notebook_path',
            'notebook': 'notebook_path',
            'cell': 'cell_number',
            'index': 'cell_number',
            'cell_index': 'cell_number',
            'source': 'new_source',
            'content': 'new_source',
            'cell_content': 'new_source',
            'mode': 'edit_mode',
            'type': 'cell_type'
        },
        'GlobTool': {
            'directory': 'path',
            'dir': 'path',
            'search_path': 'path',
            'glob': 'pattern',
            'search': 'pattern',
            'query': 'pattern'
        },
        'GrepTool': {
            'directory': 'path',
            'dir': 'path',
            'search_path': 'path',
            'search': 'pattern',
            'query': 'pattern',
            'text': 'pattern',
            'regex': 'pattern',
            'file_type': 'include',
            'extension': 'include',
            'file_pattern': 'include'
        },
        'LS': {
            'directory': 'path',
            'dir': 'path',
            'folder': 'path',
            'exclude': 'ignore',
            'excluded': 'ignore',
            'ignore_patterns': 'ignore'
        },
        'WebFetchTool': {
            'website': 'url',
            'link': 'url',
            'uri': 'url',
            'query': 'prompt',
            'question': 'prompt',
            'instruction': 'prompt'
        },
        'dispatch_agent': {
            'task': 'prompt',
            'instruction': 'prompt',
            'query': 'prompt',
            'question': 'prompt'
        },
        'BatchTool': {
            'description': 'description',
            'calls': 'invocations',
            'tools': 'invocations'
        }
    }
    
    # Apply mappings if tool name exists in our mapping dictionary
    if name in parameter_mappings:
        mappings = parameter_mappings[name]
        updated_arguments = {}
        
        # Copy all existing arguments
        for key, value in arguments.items():
            # If this key should be mapped to another name
            if key in mappings:
                mapped_key = mappings[key]
                updated_arguments[mapped_key] = value
                logger.warning(f"PARAMETER MAP: Mapped tool parameter {key} -> {mapped_key} for tool {name}")
            else:
                updated_arguments[key] = value
                
        # Special handling for Edit tool if parameters are missing
        if name == 'Edit':
            # Create a blank old_string if we're doing a file creation operation
            if 'file_path' in updated_arguments and 'new_string' in updated_arguments and 'old_string' not in updated_arguments:
                updated_arguments['old_string'] = ''
                logger.warning(f"PARAMETER MAP: Added empty old_string parameter for Edit tool with file_path and new_string")
            elif 'path' in arguments and 'content' in arguments and 'old_string' not in updated_arguments:
                # Check if we need to map more parameters
                if 'path' in arguments and 'path' not in mappings:
                    updated_arguments['file_path'] = arguments['path']
                if 'content' in arguments and 'content' not in mappings:
                    updated_arguments['new_string'] = arguments['content']
                updated_arguments['old_string'] = ''
                logger.warning(f"PARAMETER MAP: Added empty old_string parameter for Edit tool with path and content")
        
        # Special handling for Replace tool if parameters are missing
        elif name == 'Replace':
            if 'file_path' in updated_arguments and 'content' not in updated_arguments:
                # Try to find content in other parameters
                for param in ['text', 'new_content', 'data']:
                    if param in arguments:
                        updated_arguments['content'] = arguments[param]
                        logger.warning(f"PARAMETER MAP: Mapped {param} to content for Replace tool")
                        break
        
        # Special handling for View tool if parameters are missing expected format
        elif name == 'View':
            # Convert string offsets and limits to integers
            for param in ['offset', 'limit']:
                if param in updated_arguments and isinstance(updated_arguments[param], str):
                    try:
                        updated_arguments[param] = int(updated_arguments[param])
                        logger.warning(f"PARAMETER MAP: Converted {param} from string to integer for View tool")
                    except ValueError:
                        logger.warning(f"PARAMETER MAP: Could not convert {param} value '{updated_arguments[param]}' to integer")
        
        # Special handling for NotebookEditCell tool
        elif name == 'NotebookEditCell':
            # Convert cell_number from string to integer if needed
            if 'cell_number' in updated_arguments and isinstance(updated_arguments['cell_number'], str):
                try:
                    updated_arguments['cell_number'] = int(updated_arguments['cell_number'])
                    logger.warning(f"PARAMETER MAP: Converted cell_number from string to integer for NotebookEditCell tool")
                except ValueError:
                    logger.warning(f"PARAMETER MAP: Could not convert cell_number value '{updated_arguments['cell_number']}' to integer")
        
        # Update the arguments
        function_call['arguments'] = updated_arguments
        
    return tool_call

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Special handling for Gemini model - need to modify some parameters
    is_gemini = False
    if anthropic_request.model and "gemini" in anthropic_request.model:
        is_gemini = True
        logger.info(f"Special handling for Gemini model in request conversion")
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            # Format the tool result differently for Gemini vs. other models
                            if is_gemini:
                                # Gemini prefers simpler tool result format
                                text_content += f"Tool result: {result_content}\n"
                            else:
                                text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Special handling for Gemini models - they use a different format for tool calls
                            if is_gemini:
                                # For Gemini, convert to functionCall format
                                try:
                                    # Format arguments as a JSON string if they're not already
                                    args = block.input if hasattr(block, "input") else {}
                                    if isinstance(args, dict):
                                        args_str = json.dumps(args)
                                    else:
                                        args_str = str(args)
                                    
                                    processed_content.append({
                                        "type": "function_call",
                                        "function_call": {
                                            "name": block.name if hasattr(block, "name") else "unknown_tool",
                                            "arguments": args_str
                                        }
                                    })
                                    logger.info(f"Transformed tool_use to Gemini function_call format")
                                except Exception as e:
                                    logger.warning(f"Error converting tool_use to Gemini format: {str(e)}")
                                    # Fallback to standard format
                                    processed_content.append({
                                        "type": "tool_use",
                                        "id": block.id if hasattr(block, "id") else f"call_{uuid.uuid4()}",
                                        "name": block.name if hasattr(block, "name") else "unknown_tool",
                                        "input": block.input if hasattr(block, "input") else {}
                                    })
                            else:
                                # For other models, use standard format
                                processed_content.append({
                                    "type": "tool_use",
                                    "id": block.id if hasattr(block, "id") else f"call_{uuid.uuid4()}",
                                    "name": block.name if hasattr(block, "name") else "unknown_tool",
                                    "input": block.input if hasattr(block, "input") else {}
                                })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens for Deepseek and Gemini models to their limit of 8192
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("deepseek/") or anthropic_request.model.startswith("gemini/") or USE_OPENAI_MODELS:
        max_tokens = min(max_tokens, 8192)
        logger.debug(f"Capping max_tokens to 8192 for {anthropic_request.model} (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    # Add thinking parameter if present (newer Claude API feature)
    if anthropic_request.thinking:
        # For OpenAI-compatible APIs like Deepseek, we include this in metadata
        # Since they don't directly support the thinking parameter
        if not "metadata" in litellm_request:
            litellm_request["metadata"] = {}
            
        # Handle different formats of the thinking parameter
        if isinstance(anthropic_request.thinking, dict) and "enabled" in anthropic_request.thinking:
            litellm_request["metadata"]["thinking"] = {
                "enabled": bool(anthropic_request.thinking["enabled"])
            }
        else:
            # Default to enabled=True if thinking is present but format is unexpected
            litellm_request["metadata"]["thinking"] = {
                "enabled": True
            }
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        logger.warning(f"TOOL DEBUG: Converting {len(anthropic_request.tools)} tools to OpenAI format")
        openai_tools = []
        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'model_dump'):
                tool_dict = tool.model_dump()
            elif hasattr(tool, 'dict'):
                # For backward compatibility with older Pydantic
                tool_dict = tool.dict()
            else:
                tool_dict = tool
                
            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": tool_dict["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
            
        litellm_request["tools"] = openai_tools
        
        # WORKAROUND: For Deepseek model, try forcing tool use
        # If the model is deepseek and we have tools, force the model to use a specific tool
        if "deepseek" in anthropic_request.model.lower():
            # Get the first tool
            first_tool_name = openai_tools[0]["function"]["name"]
            logger.warning(f"TOOL DEBUG: WORKAROUND - Forcing Deepseek to use tool: {first_tool_name}")
            
            # Set tool_choice to force using this function
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": first_tool_name}
            }
            logger.warning(f"TOOL DEBUG: Set tool_choice to force function: {first_tool_name}")
        
        logger.warning(f"TOOL DEBUG: Converted tools: {json.dumps(openai_tools)}")
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'model_dump'):
            tool_choice_dict = anthropic_request.tool_choice.model_dump()
        elif hasattr(anthropic_request.tool_choice, 'dict'):
            # For backward compatibility with older Pydantic
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("deepseek/"):
            clean_model = clean_model[len("deepseek/"):]
        elif clean_model.startswith("gemini/"):
            clean_model = clean_model[len("gemini/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            
            # Apply parameter mapping to fix common parameter naming issues
            if tool_calls:
                if isinstance(tool_calls, list):
                    # Map parameters for each tool call
                    mapped_tool_calls = []
                    for tc in tool_calls:
                        # Handle different object types
                        if hasattr(tc, 'function'):
                            # It's a direct object with function attribute
                            func_obj = tc.function
                            tool_id = getattr(tc, 'id', f"tool_{uuid.uuid4()}")
                            
                            # Get function name
                            name = ""
                            if hasattr(func_obj, 'name'):
                                name = func_obj.name
                                
                            # Get function arguments
                            args = {}
                            if hasattr(func_obj, 'arguments'):
                                args_str = func_obj.arguments
                                if isinstance(args_str, str):
                                    try:
                                        args = json.loads(args_str)
                                    except json.JSONDecodeError:
                                        args = {"raw_args": args_str}
                                else:
                                    args = args_str
                                    
                            # Create a properly structured tool call
                            tc_dict = {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args
                                }
                            }
                            
                            logger.warning(f"TOOL DEBUG: Created structured tool call from object: {json.dumps(tc_dict, default=str)}")
                        elif isinstance(tc, dict):
                            # It's already a dict
                            tc_dict = tc
                            logger.warning(f"TOOL DEBUG: Using existing tool call dict: {json.dumps(tc_dict, default=str)}")
                        else:
                            # Try to convert to dict
                            try:
                                if hasattr(tc, 'model_dump'):
                                    tc_dict = tc.model_dump()
                                elif hasattr(tc, 'dict'):
                                    tc_dict = tc.dict()
                                elif hasattr(tc, '__dict__'):
                                    tc_dict = tc.__dict__
                                else:
                                    # Last resort - serialize and create empty dict
                                    logger.warning(f"TOOL DEBUG: Unable to convert tool call to dict, type: {type(tc)}")
                                    tc_dict = {"raw": str(tc)}
                            except Exception as e:
                                logger.warning(f"TOOL DEBUG: Error converting tool call to dict: {str(e)}")
                                tc_dict = {"error": str(e)}
                                
                        # Now apply parameter mapping
                        mapped_tc = map_tool_parameters(tc_dict)
                        mapped_tool_calls.append(mapped_tc)
                        
                    tool_calls = mapped_tool_calls
                else:
                    # Single tool call - handle same as above
                    tc = tool_calls
                    if hasattr(tc, 'function'):
                        # It's a direct object with function attribute
                        func_obj = tc.function
                        tool_id = getattr(tc, 'id', f"tool_{uuid.uuid4()}")
                        
                        # Get function name
                        name = ""
                        if hasattr(func_obj, 'name'):
                            name = func_obj.name
                            
                        # Get function arguments
                        args = {}
                        if hasattr(func_obj, 'arguments'):
                            args_str = func_obj.arguments
                            if isinstance(args_str, str):
                                try:
                                    args = json.loads(args_str)
                                except json.JSONDecodeError:
                                    args = {"raw_args": args_str}
                            else:
                                args = args_str
                                
                        # Create a properly structured tool call
                        tc_dict = {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args
                            }
                        }
                        
                        logger.warning(f"TOOL DEBUG: Created structured tool call from object: {json.dumps(tc_dict, default=str)}")
                    elif isinstance(tc, dict):
                        # It's already a dict
                        tc_dict = tc
                        logger.warning(f"TOOL DEBUG: Using existing tool call dict: {json.dumps(tc_dict, default=str)}")
                    else:
                        # Try to convert to dict
                        try:
                            if hasattr(tc, 'model_dump'):
                                tc_dict = tc.model_dump()
                            elif hasattr(tc, 'dict'):
                                tc_dict = tc.dict()
                            elif hasattr(tc, '__dict__'):
                                tc_dict = tc.__dict__
                            else:
                                # Last resort - serialize and create empty dict
                                logger.warning(f"TOOL DEBUG: Unable to convert tool call to dict, type: {type(tc)}")
                                tc_dict = {"raw": str(tc)}
                        except Exception as e:
                            logger.warning(f"TOOL DEBUG: Error converting tool call to dict: {str(e)}")
                            tc_dict = {"error": str(e)}
                            
                    # Apply parameter mapping
                    tool_calls = map_tool_parameters(tc_dict)
                    
                logger.warning(f"TOOL DEBUG: After parameter mapping: {json.dumps(tool_calls, default=str)}")
                
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            
            # Apply parameter mapping to fix common parameter naming issues
            if tool_calls:
                if isinstance(tool_calls, list):
                    # Map parameters for each tool call
                    mapped_tool_calls = []
                    for tc in tool_calls:
                        mapped_tc = map_tool_parameters(tc)
                        mapped_tool_calls.append(mapped_tc)
                    tool_calls = mapped_tool_calls
                else:
                    # Single tool call
                    tool_calls = map_tool_parameters(tool_calls)
                    
                logger.warning(f"TOOL DEBUG: After parameter mapping: {json.dumps(tool_calls, default=str)}")
                
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            
            # Add debug logging for finish_reason
            if finish_reason == "tool_calls":
                logger.warning(f"TOOL DEBUG: Response has finish_reason='tool_calls', indicating tool usage")
            elif tool_calls:
                logger.warning(f"TOOL DEBUG: Response has tool_calls but finish_reason='{finish_reason}'")
                
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # FEATURE: If tool_calls is empty/None but the content_text suggests a tool call,
        # try to extract it from the text response
        if (not tool_calls or tool_calls == []) and content_text and ("deepseek" in clean_model.lower() or "gemini" in clean_model.lower()):
            # Apply this logic to Deepseek and Gemini models which might not support function calling properly
            logger.warning(f"TOOL DEBUG: Attempting to extract tool calls from text: {content_text[:100]}...")
            
            # Check for sequence indicators that might suggest multiple tools will be used
            multiple_tool_sequence = False
            sequence_markers = [
                "I'll follow these steps:",
                "Here are the steps I'll follow:",
                "I'll perform the following actions:",
                "Step 1:",
                "First, I'll",
                "First I'll use",
                "First, I will use",
                "Let me use multiple tools:",
                "I need to use several tools:",
                "I'll use the following tools in sequence:",
                "To complete this task, I'll use:"
            ]
            
            for marker in sequence_markers:
                if marker in content_text:
                    multiple_tool_sequence = True
                    logger.warning(f"TOOL DEBUG: Detected potential multiple tool sequence with marker: '{marker}'")
                    break
            
            # Common patterns for tool usage in text
            tool_patterns = [
                # Format: "Using the X tool with parameters..."
                r"(?:Using|Use|I'll use|Let me use|Using the|I will use)(?: the)? ([A-Za-z]+)(?: tool)? (?:with|to|for)(.*?)(?:\.|\n|$)",
                
                # Format: "[Tool: X (arguments)]"
                r"\[Tool: ([A-Za-z]+)(?: \(([^)]*)\))?\]",
                
                # Format: "Tool(param1=value, param2=value)"
                r"([A-Za-z]+)\(([^)]*)\)",
                
                # Format: "Tool: X, Parameters: {...}"
                r"Tool: ([A-Za-z]+)(?:,|\n|\s+)(?:Parameters|Arguments|Params|Args|Input)?: (.+?)(?:\.|\n|$)",
                
                # Format: "I need to use the X tool to..."
                r"(?:I need to|I should|I can|I must|We should|We need to|Let's)(?: use| execute| call| run| invoke)(?: the)? ([A-Za-z]+)(?: tool| function)?(.*?)(?:\.|\n|$)",
                
                # Format: "Execute X with parameters..."
                r"(?:Execute|Run|Call|Invoke|Utilize|Apply)(?: the)? ([A-Za-z]+)(?: tool| function)?(.*?)(?:\.|\n|$)",
                
                # Format: "To accomplish this, I'll use X..."
                r"(?:To|For)(?: this| that)(?:,|:)? (?:I'll|I will|we'll|we will|I'm going to|I am going to|let me|let's) (?:use|try|call|execute|run|utilize) (?:the )?([A-Za-z]+)(?: tool| function)?(.*?)(?:\.|\n|$)",
                
                # Format: "X tool can be used with parameters..."
                r"(?:The )?([A-Za-z]+)(?: tool| function) (?:can be|should be|will be|is) (?:used|executed|called|run|invoked|applied)(.*?)(?:\.|\n|$)"
            ]
            
            extracted_tool = None
            extracted_tools = []  # Keep track of all tools to handle multiple tool sequences
            
            for pattern in tool_patterns:
                matches = re.findall(pattern, content_text, re.DOTALL | re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            tool_name = match[0].strip()
                            param_text = match[1].strip()
                            
                            # Store information about this potential tool
                            extracted_tools.append({
                                "tool_name": tool_name,
                                "param_text": param_text,
                                "pattern": pattern
                            })
            
            # If we found multiple potential tool matches
            if len(extracted_tools) > 0:
                logger.warning(f"TOOL DEBUG: Found {len(extracted_tools)} potential tool references in text")
                
                # For multiple tool sequences, just use the first valid tool
                for potential_tool in extracted_tools:
                    tool_name = potential_tool["tool_name"]
                    param_text = potential_tool["param_text"]
                    
                    # Check if tool name is one of our supported tools
                    supported_tools = ["Bash", "BatchTool", "GlobTool", "GrepTool", "LS", 
                                      "View", "Edit", "Replace", "ReadNotebook", 
                                      "NotebookEditCell", "WebFetchTool", "dispatch_agent"]
                    
                    if tool_name in supported_tools:
                        logger.warning(f"TOOL DEBUG: Extracted tool '{tool_name}' from text with params: {param_text}")
                        
                        # Try to parse parameters
                        params = {}
                        
                        # First, check if the param_text looks like valid JSON
                        try:
                            # Check if it starts with '{' and ends with '}'
                            if param_text.strip().startswith('{') and param_text.strip().endswith('}'):
                                json_params = json.loads(param_text.strip())
                                if isinstance(json_params, dict):
                                    params = json_params
                                    logger.warning(f"TOOL DEBUG: Successfully extracted parameters as JSON: {params}")
                        except Exception as e:
                            logger.warning(f"TOOL DEBUG: JSON parsing failed: {str(e)}, falling back to regex parsing")
                        
                        # If JSON parsing failed or no parameters were found, try regex patterns
                        if not params:
                            # Try multiple regex patterns for different parameter formats
                            
                            # Pattern 1: key="value" or key='value' or key=value
                            param_pairs = re.findall(r'(\w+)\s*[=:]\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]*))', param_text)
                            for pair in param_pairs:
                                key = pair[0]
                                # Find first non-empty value
                                value = next((v for v in pair[1:] if v), "")
                                params[key] = value
                            
                            # Pattern 2: "key": "value" (JSON-like but without braces)
                            if not params:
                                param_pairs = re.findall(r'"(\w+)"\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]*))', param_text)
                                for pair in param_pairs:
                                    key = pair[0]
                                    # Find first non-empty value
                                    value = next((v for v in pair[1:] if v), "")
                                    params[key] = value
                            
                            # Pattern 3: key: value (YAML-like)
                            if not params:
                                param_pairs = re.findall(r'(\w+)\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\n]*))', param_text)
                                for pair in param_pairs:
                                    key = pair[0]
                                    # Find first non-empty value
                                    value = next((v for v in pair[1:] if v), "").strip()
                                    params[key] = value
                            
                            # Pattern 4: key is value
                            if not params:
                                param_pairs = re.findall(r'(\w+)\s+is\s+(?:"([^"]*)"|\'([^\']*)\'|([^,\n]*))', param_text)
                                for pair in param_pairs:
                                    key = pair[0]
                                    # Find first non-empty value
                                    value = next((v for v in pair[1:] if v), "").strip()
                                    params[key] = value
                            
                            logger.warning(f"TOOL DEBUG: Extracted parameters via regex: {params}")
                        
                        # Create a synthetic tool call
                        extracted_tool = {
                            "id": f"extracted_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": params
                            }
                        }
                        
                        # Apply parameter mapping to fix any parameter naming issues
                        extracted_tool = map_tool_parameters(extracted_tool)
                        
                        # Remove the tool usage part from content_text to avoid duplication
                        match_text = potential_tool["tool_name"] + potential_tool["param_text"]
                        content_text = content_text.replace(match_text, "").strip()
                        
                        # Log the extracted tool
                        logger.warning(f"TOOL DEBUG: Constructed synthetic tool call: {json.dumps(extracted_tool)}")
                        break
                    
                # No break needed here - we'll use the extracted_tool if found
            
            # If we found a tool call, add it to the tool_calls list
            if extracted_tool:
                tool_calls = [extracted_tool]
                # Set finish_reason to tool_calls to indicate tool usage
                finish_reason = "tool_calls"
                logger.warning(f"TOOL DEBUG: Added synthetic tool call to response")
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format)
        if tool_calls:
            # Enhanced logging for tool calls
            try:
                if isinstance(tool_calls, list):
                    logger.warning(f"TOOL DEBUG: Received {len(tool_calls)} tool calls from model: {json.dumps(tool_calls)}")
                else:
                    logger.warning(f"TOOL DEBUG: Received tool call from model: {json.dumps([tool_calls])}")
            except Exception as e:
                logger.warning(f"TOOL DEBUG: Error serializing tool calls: {str(e)}, raw tool_calls: {tool_calls}")
                
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            
            # For any model with valid tool calls, format as tool_use blocks
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                # Add the tool_use content block
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Check if we're using Gemini model for special handling
        is_gemini = False
        if hasattr(original_request, 'model') and "gemini" in original_request.model:
            is_gemini = True
            logger.info(f"Using special Gemini handling for streaming responses")
            
            # Check if tools are being used with Gemini
            if hasattr(original_request, 'tools') and original_request.tools:
                tool_count = len(original_request.tools)
                logger.info(f"Gemini streaming with {tool_count} tools - using enhanced function call processing")
                
                # Set some markers for tracking function call state
                function_call_accumulator = None
                is_expecting_function_call = True if original_request.tool_choice != "none" else False
                function_call_complete = False
        
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.original_model if hasattr(original_request, 'original_model') else original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    """Main endpoint for message creation - handles special commands like /brainstorm."""
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Check for custom commands in the user message
        if "messages" in body_json and body_json["messages"]:
            for i, msg in enumerate(body_json["messages"]):
                if msg.get("role") == "user":
                    # Check for commands in string content
                    if isinstance(msg.get("content"), str):
                        content = msg.get("content", "")
                        # Check for /brainstorm command
                        if content.strip().startswith("/brainstorm "):
                            logger.info(f"🧠 Detected /brainstorm command - redirecting to brainstorm endpoint")
                            # Extract the query (everything after /brainstorm )
                            query = content.strip()[12:].strip()
                            # Create a new message with just the query
                            body_json["messages"][i]["content"] = query
                            # Forward to brainstorm endpoint
                            return await brainstorm(raw_request)
                    
                    # Check for commands in content blocks
                    elif isinstance(msg.get("content"), list):
                        for j, block in enumerate(msg["content"]):
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                # Check for /brainstorm command
                                if text.strip().startswith("/brainstorm "):
                                    logger.info(f"🧠 Detected /brainstorm command - redirecting to brainstorm endpoint")
                                    # Extract the query (everything after /brainstorm )
                                    query = text.strip()[12:].strip()
                                    # Create a new block with just the query
                                    body_json["messages"][i]["content"][j]["text"] = query
                                    # Forward to brainstorm endpoint
                                    return await brainstorm(raw_request)
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("deepseek/"):
            clean_model = clean_model[len("deepseek/"):]
        elif clean_model.startswith("gemini/"):
            clean_model = clean_model[len("gemini/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine which API key to use based on the model
        if request.model.startswith("deepseek/"):
            litellm_request["api_key"] = DEEPSEEK_API_KEY
            logger.debug(f"Using Deepseek API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            # Use key rotation system for Gemini API calls
            litellm_request["api_key"] = get_next_gemini_api_key()
            logger.debug(f"Using Gemini API key #{gemini_key_counter+1}/{len(GEMINI_API_KEYS)} for model: {request.model}")
            
            # Clean tool schemas for Gemini compatibility
            if "tools" in litellm_request:
                original_tools = litellm_request["tools"]
                cleaned_tools = clean_gemini_tools(original_tools)
                litellm_request["tools"] = cleaned_tools
                logger.debug(f"Cleaned tool schemas for Gemini compatibility, tools count: {len(cleaned_tools)}")
                
            # Fix tool_choice for Gemini
            if "tool_choice" in litellm_request:
                logger.info(f"Original tool_choice: {json.dumps(litellm_request['tool_choice'])}")
                
                # Gemini expects either "none", "auto", or a specific function
                # Auto in Gemini is called "any" (different from OpenAI's "auto")
                if isinstance(litellm_request["tool_choice"], str):
                    if litellm_request["tool_choice"] == "auto":
                        litellm_request["tool_choice"] = "any"
                        logger.info(f"Changed tool_choice from 'auto' to 'any' for Gemini compatibility")
                    elif litellm_request["tool_choice"] == "required":
                        # Force tool use
                        litellm_request["tool_choice"] = "any"
                        logger.info(f"Changed tool_choice from 'required' to 'any' for Gemini compatibility")
                
                # If it's a dict with function object, ensure it's in the right format
                elif isinstance(litellm_request["tool_choice"], dict):
                    # Handle the "function" key case
                    if "function" in litellm_request["tool_choice"]:
                        # Extract the function name to force that specific function
                        function_name = litellm_request["tool_choice"]["function"].get("name")
                        if function_name:
                            # For specific named function, Gemini needs a specific format
                            litellm_request["tool_choice"] = {
                                "type": "function",
                                "function": {"name": function_name}
                            }
                            logger.info(f"Reformatted function tool_choice to Gemini format for function: {function_name}")
                        else:
                            # If no specific function, default to any
                            litellm_request["tool_choice"] = "any"
                            logger.info(f"Simplified function tool_choice without name to 'any' for Gemini compatibility")
                    else:
                        # For all other cases, simply use 'any'
                        litellm_request["tool_choice"] = "any"
                        logger.info(f"Simplified complex tool_choice to 'any' for Gemini compatibility")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        
        # For Deepseek or Gemini models - modify request format to work with limitations
        if ("deepseek" in litellm_request["model"] or "gemini" in litellm_request["model"]) and "messages" in litellm_request:
            logger.debug(f"Processing {litellm_request['model']} model request")
            
            # For Deepseek models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Log additional details about tool usage in request
            if num_tools > 0:
                logger.warning(f"TOOL DEBUG: Making completion request with {num_tools} tools enabled, model={litellm_request.get('model')}")
                if 'tool_choice' in litellm_request:
                    logger.warning(f"TOOL DEBUG: tool_choice set to: {litellm_request.get('tool_choice')}")
                
                # Log detailed request information
                try:
                    # Log messages
                    messages_count = len(litellm_request.get('messages', []))
                    logger.warning(f"REQUEST DEBUG: Sending {messages_count} messages")
                    
                    # Log last user message 
                    last_user_message = None
                    for msg in reversed(litellm_request.get('messages', [])):
                        if msg.get('role') == 'user':
                            last_user_message = msg
                            break
                            
                    if last_user_message:
                        logger.warning(f"REQUEST DEBUG: Last user message: {json.dumps(last_user_message.get('content'), indent=2)[:500]}...")
                    
                    # Log tools details
                    tools = litellm_request.get('tools', [])
                    tool_names = [tool.get('function', {}).get('name') for tool in tools]
                    logger.warning(f"REQUEST DEBUG: Tool names: {tool_names}")
                    
                    # Log detailed tool information for easier debugging
                    logger.warning(f"TOOL DEBUG: Sending {len(tools)} tool definitions to model")
                    for i, tool in enumerate(tools):
                        if 'function' in tool:
                            tool_name = tool['function'].get('name', 'unnamed')
                            tool_desc = tool['function'].get('description', '')[:100]
                            logger.warning(f"TOOL DEBUG: Tool {i+1}: {tool_name} - {tool_desc}")
                            
                            # Log parameter schema
                            if 'parameters' in tool['function']:
                                param_schema = tool['function']['parameters']
                                required_params = param_schema.get('required', [])
                                properties = param_schema.get('properties', {})
                                
                                param_summary = []
                                for param_name, param_info in properties.items():
                                    is_required = "Required" if param_name in required_params else "Optional"
                                    param_type = param_info.get('type', 'unknown')
                                    param_desc = param_info.get('description', '')[:50]
                                    param_summary.append(f"{param_name} ({param_type}, {is_required}): {param_desc}")
                                
                                logger.warning(f"TOOL DEBUG: {tool_name} parameters: {json.dumps(param_summary, indent=2)}")
                    
                    # Log a summary of the request
                    req_summary = {
                        "model": litellm_request.get('model'),
                        "messages_count": messages_count,
                        "tools_count": len(tools),
                        "tool_names": tool_names,
                        "max_tokens": litellm_request.get('max_tokens'),
                        "temperature": litellm_request.get('temperature'),
                        "has_tool_choice": 'tool_choice' in litellm_request
                    }
                    logger.warning(f"REQUEST DEBUG: Request summary: {json.dumps(req_summary, indent=2)}")
                    
                except Exception as e:
                    logger.warning(f"REQUEST DEBUG: Error logging request details: {str(e)}")
            
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Detailed logging of raw response for debugging
            try:
                # Get provider from model
                provider = "unknown"
                if "deepseek" in litellm_request.get('model', ""):
                    provider = "deepseek"
                elif "gemini" in litellm_request.get('model', ""):
                    provider = "gemini"
                    
                # Save raw string representation for debugging
                raw_str = str(litellm_response)
                logger.warning(f"ULTRA RAW RESPONSE DEBUG [{provider}]: {raw_str[:2000]}")
                
                # Convert to dict if it's not already
                if hasattr(litellm_response, 'dict'):
                    raw_response = litellm_response.dict()
                elif hasattr(litellm_response, 'model_dump'):
                    raw_response = litellm_response.model_dump()
                else:
                    raw_response = litellm_response
                
                # Get direct access to the raw response object
                direct_response = None
                if hasattr(litellm_response, '_response'):
                    direct_response = litellm_response._response
                    logger.warning(f"DIRECT RESPONSE DEBUG [{provider}]: {str(direct_response)[:2000]}")
                    
                    # Try to get JSON from direct response
                    try:
                        if hasattr(direct_response, 'json'):
                            direct_json = direct_response.json()
                            logger.warning(f"DIRECT JSON RESPONSE DEBUG [{provider}]: {json.dumps(direct_json, indent=2)[:2000]}")
                    except Exception as json_e:
                        logger.warning(f"Error extracting JSON from direct response: {str(json_e)}")
                
                # Convert to string for logging
                try:
                    # Use custom serializer to handle non-serializable objects
                    def default_serializer(obj):
                        try:
                            return str(obj)
                        except:
                            return "<non-serializable>"
                            
                    raw_response_str = json.dumps(raw_response, indent=2, default=default_serializer)
                    logger.warning(f"RAW RESPONSE DEBUG: Full response from {litellm_request.get('model')}:\n{raw_response_str[:5000]}")
                except Exception as json_err:
                    logger.warning(f"Error serializing response to JSON: {str(json_err)}")
                    logger.warning(f"Falling back to str representation: {str(raw_response)[:2000]}")
                
                # Log choices specifically for easier analysis
                if hasattr(litellm_response, 'choices'):
                    choices = litellm_response.choices
                    logger.warning(f"RAW RESPONSE DEBUG: Number of choices: {len(choices)}")
                    for i, choice in enumerate(choices):
                        logger.warning(f"RAW RESPONSE DEBUG: Choice {i}:")
                        
                        # Log all attributes of choice
                        for attr_name in dir(choice):
                            if not attr_name.startswith('_'):
                                try:
                                    attr_value = getattr(choice, attr_name)
                                    if not callable(attr_value):
                                        logger.warning(f"RAW RESPONSE DEBUG: choice.{attr_name} = {attr_value}")
                                except Exception as attr_err:
                                    logger.warning(f"Error getting attr {attr_name}: {str(attr_err)}")
                        
                        # Log finish_reason
                        finish_reason = getattr(choice, 'finish_reason', None)
                        logger.warning(f"RAW RESPONSE DEBUG: finish_reason = {finish_reason}")
                        
                        # Log message
                        message = getattr(choice, 'message', None)
                        if message:
                            # Log all attributes of message
                            for msg_attr in dir(message):
                                if not msg_attr.startswith('_'):
                                    try:
                                        msg_attr_value = getattr(message, msg_attr)
                                        if not callable(msg_attr_value):
                                            logger.warning(f"RAW RESPONSE DEBUG: message.{msg_attr} = {msg_attr_value}")
                                    except Exception as msg_attr_err:
                                        logger.warning(f"Error getting message attr {msg_attr}: {str(msg_attr_err)}")
                            
                            # Log content
                            content = getattr(message, 'content', None)
                            logger.warning(f"RAW RESPONSE DEBUG: content type = {type(content)}, value = {content}")
                            
                            # Log role
                            role = getattr(message, 'role', None)
                            logger.warning(f"RAW RESPONSE DEBUG: role = {role}")
                            
                            # Log tool_calls specifically
                            tool_calls = getattr(message, 'tool_calls', None)
                            if tool_calls:
                                logger.warning(f"RAW RESPONSE DEBUG: tool_calls present = TRUE, type = {type(tool_calls)}")
                                if isinstance(tool_calls, list):
                                    logger.warning(f"RAW RESPONSE DEBUG: Number of tool_calls: {len(tool_calls)}")
                                    for j, call in enumerate(tool_calls):
                                        try:
                                            call_str = json.dumps(call, default=default_serializer)
                                            logger.warning(f"RAW RESPONSE DEBUG: Tool call {j}: {call_str}")
                                        except:
                                            logger.warning(f"RAW RESPONSE DEBUG: Tool call {j}: {str(call)}")
                                else:
                                    try:
                                        tool_calls_str = json.dumps(tool_calls, default=default_serializer)
                                        logger.warning(f"RAW RESPONSE DEBUG: Single tool_call: {tool_calls_str}")
                                    except:
                                        logger.warning(f"RAW RESPONSE DEBUG: Single tool_call: {str(tool_calls)}")
                            else:
                                logger.warning(f"RAW RESPONSE DEBUG: tool_calls present = FALSE")
            except Exception as e:
                logger.warning(f"RAW RESPONSE DEBUG: Error logging response: {str(e)}")
                # Try a simpler approach
                try:
                    logger.warning(f"RAW RESPONSE DEBUG: Simple response dump: {str(litellm_response)}")
                except Exception as e2:
                    logger.warning(f"RAW RESPONSE DEBUG: Failed to log even simple response: {str(e2)}")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Handle 'response' attribute specially since it might not be JSON serializable
        if hasattr(e, 'response'):
            try:
                # Check if it's a response object
                if hasattr(e.response, 'status_code') and hasattr(e.response, 'text'):
                    # It's likely a httpx.Response object
                    error_details['response'] = {
                        'status_code': getattr(e.response, 'status_code', None),
                        'text': str(getattr(e.response, 'text', ''))
                    }
                else:
                    # Try to convert to string
                    error_details['response'] = str(e.response)
            except:
                # If all else fails, just note that there was a response
                error_details['response'] = "Response object (not serializable)"
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__', 'response']:
                    try:
                        # Try to convert to string to avoid serialization issues
                        error_details[key] = str(value)
                    except:
                        error_details[key] = f"<non-serializable {type(value).__name__}>"
        
        # Log all error details
        try:
            logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        except TypeError:
            # Fallback if JSON serialization fails
            logger.error(f"Error processing request: {error_details['type']}: {error_details['error']}")
            logger.error(f"Full traceback: {error_traceback}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if hasattr(e, 'message') and getattr(e, 'message'):
            error_message += f"\nMessage: {getattr(e, 'message')}"
        if hasattr(e, 'response'):
            try:
                if hasattr(e.response, 'text'):
                    error_message += f"\nResponse: {getattr(e.response, 'text', '')}"
                else:
                    error_message += f"\nResponse: {str(e.response)}"
            except:
                error_message += "\nResponse: <non-serializable response object>"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("deepseek/"):
            clean_model = clean_model[len("deepseek/"):]
        
        elif clean_model.startswith("gemini/"):
            clean_model = clean_model[len("gemini/"):]
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Count tokens
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

# Specialized prompt templates
BRAINSTORM_PROMPT = """I want you to act as a code expert brainstorming assistant. Your goal is to help generate diverse, creative, and actionable ideas for improving or solving coding problems.

For the user's input topic or question about code, please:
1. Generate at least 5 distinct ideas or approaches
2. For each idea, provide:
   - A concise title or summary
   - A brief explanation (1-2 sentences)
   - At least one specific example, implementation detail, or code snippet
   - Any potential tradeoffs or considerations
3. Include a mix of conventional and innovative solutions
4. Consider different perspectives, constraints, and opportunities
5. Leverage best practices and modern development techniques
6. Focus on providing actionable, practical advice for implementation

Present the ideas in a clear, structured format. Focus on being comprehensive yet concise.

User's topic: {user_input}
"""

@app.post("/v1/brainstorm")
async def brainstorm(
    raw_request: Request
):
    """Custom endpoint for brainstorming - enhances prompts with a specialized system prompt."""
    try:
        # Get the raw request body
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        
        # Log the incoming brainstorm request
        logger.debug(f"📊 PROCESSING BRAINSTORM REQUEST")
        
        # Extract user query from messages
        user_input = ""
        if "messages" in body_json and body_json["messages"]:
            for msg in body_json["messages"]:
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), str):
                        user_input = msg["content"]
                        logger.debug(f"📌 Extracted user input: {user_input[:50]}...")
                        break
                    elif isinstance(msg.get("content"), list):
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                user_input = block.get("text", "")
                                logger.debug(f"📌 Extracted user input (from block): {user_input[:50]}...")
                                break
        
        if not user_input:
            return JSONResponse(
                status_code=400,
                content={"error": "No user message found in request"}
            )
        
        # Create a new request with the brainstorming system prompt
        enhanced_request = body_json.copy()
        
        # Replace or add the system prompt
        enhanced_request["system"] = BRAINSTORM_PROMPT.format(user_input=user_input)
        
        # Use Claude 3.7 Sonnet directly from Anthropic for this special command
        enhanced_request["model"] = "claude-3-7-sonnet-20250219"
        
        # Check if Anthropic API key is available
        if not ANTHROPIC_API_KEY:
            logger.error("No Anthropic API key found for /brainstorm command - this command requires access to Claude 3.7")
            return JSONResponse(
                status_code=500,
                content={"error": "The /brainstorm command requires an Anthropic API key. Please set ANTHROPIC_API_KEY in your .env file."}
            )
            
        logger.info(f"🔄 Using Claude 3.7 Sonnet directly from Anthropic API for /brainstorm command")
        
        # Prepare the Anthropic request
        anthropic_request = {
            "model": enhanced_request["model"],
            "max_tokens": enhanced_request.get("max_tokens", 1500),
            "messages": enhanced_request["messages"],
            "system": enhanced_request["system"],
            "stream": enhanced_request.get("stream", False)
        }
        
        # Add any optional parameters
        for param in ["temperature", "top_p", "top_k"]:
            if param in enhanced_request:
                anthropic_request[param] = enhanced_request[param]
                
        # Prepare Anthropic API headers
        anthropic_headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        # Process the request directly through Anthropic API
        if anthropic_request.get("stream", False):
            async def stream_anthropic_response():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST", 
                        ANTHROPIC_API_URL, 
                        json=anthropic_request,
                        headers=anthropic_headers,
                        timeout=60
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            logger.error(f"Error from Anthropic API: {error_text.decode('utf-8')}")
                            error_json = {"error": f"Anthropic API error: {response.status_code}"}
                            yield f"data: {json.dumps(error_json)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                            
                        async for chunk in response.aiter_bytes():
                            yield chunk
            
            return StreamingResponse(
                stream_anthropic_response(),
                media_type="text/event-stream"
            )
        else:
            # Make a synchronous request to Anthropic API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    ANTHROPIC_API_URL,
                    json=anthropic_request,
                    headers=anthropic_headers,
                    timeout=60
                )
            
            if response.status_code != 200:
                logger.error(f"Error from Anthropic API: {response.text}")
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": f"Anthropic API error: {response.text}"}
                )
                
            # Simply return the Anthropic response directly
            return response.json()
    
    except Exception as e:
        logger.error(f"Error in brainstorm endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing brainstorm request: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for Deepseek and Gemini using LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, deepseek_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to Deepseek mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the target model name without provider prefix
    target_display = deepseek_model
    if "/" in target_display:
        target_display = target_display.split("/")[-1]
    
    # Color based on provider
    if "gemini" in target_display:
        target_display = f"{Colors.YELLOW}{target_display}{Colors.RESET}"
    else:
        target_display = f"{Colors.GREEN}{target_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {target_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

def print_ascii_logo():
    """Display ASCII art logo on startup."""
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    logo = f"""{BLUE}{BOLD}
████████╗███████╗███████╗██████╗ ███████╗███████╗███████╗██╗  ██╗
██╔═══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║ ██╔╝
██║   ██║█████╗  █████╗  ██████╔╝███████╗█████╗  █████╗  █████╔╝ 
██║   ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██╔══╝  ██╔══╝  ██╔═██╗ 
████████║███████╗███████╗██║     ███████║███████╗███████╗██║  ██╗
╚═══════╝╚══════╝╚══════╝╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝

{CYAN} ██████╗ ██████╗ ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║     ██║   ██║██║  ██║█████╗  
██║     ██║   ██║██║  ██║██╔══╝  
╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝{RESET}
    """
    print(logo)
    print(f"{BOLD}Proxy server for Claude Code with Deepseek and Gemini models {CYAN}(unofficial){RESET}")
    print(f"Run with: {CYAN}ANTHROPIC_BASE_URL=http://127.0.0.1:8083 claude{RESET}")
    print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8083")
        print("Optional arguments:")
        print("  --always-cot    Always add Chain-of-Thought system prompt for Sonnet models")
        sys.exit(0)
    
    # Display ASCII logo
    print_ascii_logo()
    
    # Print status info
    print(f"Starting server on http://0.0.0.0:8083")
    print(f"Chain-of-Thought mode: {'ENABLED' if ALWAYS_COT else 'DISABLED (use --always-cot to enable)'}")
    print(f"Mapping: Claude Haiku → {SMALL_MODEL}, Claude Sonnet → {BIG_MODEL}")
    print(f"Debug logging: {'ENABLED' if DEBUG_MODE else 'DISABLED (set DEBUG=true to enable)'}")
    anthropic_key_status = "AVAILABLE" if ANTHROPIC_API_KEY else "NOT AVAILABLE (required for /brainstorm)"
    print(f"Custom commands: /brainstorm (uses Claude 3.7, Anthropic API key: {anthropic_key_status})")
    print(f"Ready to process requests...\n")
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="error")