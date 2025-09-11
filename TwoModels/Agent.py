#!/usr/bin/env python
# coding: utf-8
# %%

# %%

# %%

# %%
# #!pip install langgraph
# #!pip3 install torch torchvision torchaudio transformers
# #!pip3 install packaging ninja
# #!pip3 install accelerate
# #!pip3 install protobuf
# #!pip3 install sentencepiece
# #!pip3 install bitsandbytes
# #!pip3 install scipy

import torch, os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random, json
import inspect
import json, re
from typing import Dict, Any, Optional, Callable, List

class Agent:
    def __init__(self, model_name='Qwen/Qwen2.5-Coder-7B-Instruct',
                agent_name='dummy_model', message='', asis=1, tools=[]):
        
        self.tools = dict()
        self.schema_tools = []
        self.instruct_history = 10
        self.max_iterations = 10
        self.iterations = 0
        self.max_new_tokens = 500
        self.agent_name = agent_name
        self.model_name = model_name
        
        # Initialize tools
        self.initTools(tools)
        
        # Determine save directory and precision
        if asis == 1:
            save_directory = '../'+model_name.replace('/','_')+'_saved_quality'
            torchfloat = torch.bfloat16
        else:
            save_directory = '../'+model_name.replace('/','_')+'_saved_response'
            torchfloat = torch.float16
        
        # Set up quantization config
        if asis == 1:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",  # Fixed typo from original
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.uint8,
                use_nested_quant=True,
            )
        
        # Try to load from local directory first
        try:
            print('Trying to load the model:', save_directory, 'from local repo')
            
            # Create pipeline from local model
            self.pipeline = pipeline(
                "text-generation",
                model=save_directory,
                tokenizer=save_directory,
                device_map="auto",
                torch_dtype=torchfloat,
                trust_remote_code=True
            )
            print("The requested model:", model_name, "is loaded from local")
            
        except:
            print('The model:', model_name, 'is not found locally, downloading it')
            
            # Create pipeline with quantization
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                #model_kwargs={
                #    "quantization_config": bnb_config,
                #},
                token="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",
                device_map="auto",
                torch_dtype=torchfloat,
                trust_remote_code=True
            )
            
            # Save model locally for future use
            print("Saving the model:", model_name, "locally")
            self.pipeline.model.save_pretrained(save_directory)
            self.pipeline.tokenizer.save_pretrained(save_directory)
            print("The requested model:", model_name, "is loaded and saved")
        
        # Set up pad token if needed
        if self.pipeline.tokenizer.pad_token is None:
            self.pipeline.tokenizer.add_special_tokens({'pad_token': '|PAD|'})
               
        # Initialize messages
        self.response = ""  
        if not message:
            message = "You are a helpful AI assistant. Maintain context and be concise.\n\n"
        
        if 'nstruct' in model_name:
            self.messages = [{"role": "system", "content": message}]
        else:
            self.messages = [message]
        self.message = message

    def initTools(self, tools=0):
        self.schema_tools = []
        for tool in tools:
            self.tools[tool.__name__] = tool
        for tool in self.tools:
            self.schema_tools.append(self.get_tool_schema(self.tools[tool]))

    def get_tool_schema(self, func: Callable) -> dict:
        """Generate a JSON schema for a tool function."""
        signature = inspect.signature(func)
        parameters = {}

        for name, param in signature.parameters.items():
            parameters[name] = {
                "type": "string",
                "description": f"Parameter {name}"
            }

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__.split('\n')[0] if func.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": parameters
                }
            }
        }

    def create_system_prompt(self, prompt, extras=[]):
        if 'nstruct' in self.model_name:
            return self.instruct_create_system_prompt(prompt)
        else:
            return self.llm_create_system_prompt(prompt)

    def instruct_create_system_prompt(self, prompt, extras=[]):
        if self.iterations > 0:
            return self.messages
        self.messages = [self.messages[0]]
        self.messages.append({
            "role": "user", 
            "content": prompt
        })
        return self.messages
            
    def llm_create_system_prompt(self, prompt):
        self.messages = [self.messages[0]]
        self.messages.append(prompt)
        return '\n'.join(self.messages)

    def generate_response(self, prompt, print_result='no'):
        if 'nstruct' in self.model_name:
            result = self.instruct_generate_response_pipeline(prompt)
        else:
            result = self.llm_generate_response_pipeline(prompt)
        
        if print_result == 'yes':
            print(result)
        return result

    def instruct_generate_response_pipeline(self, prompt):
        """Generate response using pipeline for instruct models with tool support."""
        
        if self.iterations > 0:
            messages = prompt
        else:
            messages = self.create_system_prompt(prompt)
        
        # For models with tool support, we might need to format manually
        # or use the pipeline's chat template
        if hasattr(self.pipeline.tokenizer, 'apply_chat_template') and self.tools:
            # Use chat template with tools
            formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tools=list(self.tools.values()) if self.tools else None,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple formatting
            formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) if hasattr(self.pipeline.tokenizer, 'apply_chat_template') else str(messages)

        # Generate using pipeline
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,  # Only return generated text
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        if isinstance(outputs, list) and len(outputs) > 0:
            generated_text = outputs[0].get('generated_text', '')
        else:
            generated_text = str(outputs)
        
        # Process tool calls if present
        jsons, _ = self.extract_all_json(generated_text)
        if len(jsons) > 0 and self.iterations < self.max_iterations:
            for json_data in jsons:
                try:
                    print('Querying tool:', json_data['name'])
                    json_result = self.tools[json_data['name']](**json_data['parameters'])
                    tool_call = {"name": json_data['name'], "arguments": {**json_data['parameters']}}
                    
                    messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
                    messages.append({"role": "tool", "name": json_data['name'], "content": json_result})
                    
                    self.iterations += 1
                    return self.instruct_generate_response_pipeline(messages)
                except Exception as e:
                    print(f'Tool call failed: {e}')
                    self.iterations = 0
                    break
            
        self.iterations = 0
        self.reset()
        self.response = generated_text
        return generated_text

    def llm_generate_response_pipeline(self, prompt):
        """Generate response using pipeline for non-instruct models."""
        messages = self.create_system_prompt(prompt)
        
        # Generate using pipeline
        outputs = self.pipeline(
            messages,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        if isinstance(outputs, list) and len(outputs) > 0:
            generated_text = outputs[0].get('generated_text', '')
        else:
            generated_text = str(outputs)
            
        return generated_text.strip()

    def extract_all_json(self, text):
        """Extract all valid JSON objects from a string."""
        json_objects = []
        text_parts = []
        try:
            matches = list(re.finditer(r"\{(?:[^{}]|{[^{}]*})*\}", text))
            if not matches:
                return [], [text.strip()]

            last_end = 0
            for match in matches:
                json_string = match.group(0)
                try:
                    data = json.loads(json_string)
                    json_objects.append(data)
                    text_parts.append(text[last_end:match.start()].strip())
                    last_end = match.end()
                except json.JSONDecodeError:
                    pass
            text_parts.append(text[last_end:].strip())

            return json_objects, text_parts

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return [], [text.strip()]

    def reset(self, message=''):
        if len(message) == 0:
            self.messages = [self.messages[0]]
        else:
            if 'nstruct' in self.model_name:
                self.messages = [{"role": "system", "content": message}]
            else:
                self.messages = [message]


# Usage example:
if __name__ == "__main__":
    # Example tool function
    def get_weather(location: str) -> str:
        """Get weather information for a location"""
        return f"The weather in {location} is sunny with 25°C"
    
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression"""
        try:
            result = eval(expression)  # Note: eval is dangerous in production
            return str(result)
        except:
            return "Invalid expression"
    
    # Create agent with tools
    agent = Agent(
        model_name='Qwen/Qwen2.5-Coder-7B-Instruct',
        agent_name='modern_agent',
        tools=[get_weather, calculate]
    )
    
    # Test the agent
    response = agent.generate_response("What's the weather like in Paris?", print_result='yes')

# %%

# %%
import json
import re

def extract_all_json(text):
    """
    Extracts all valid JSON objects from a string.

    Args:
        text: The string to search for JSON.

    Returns:
        A list of Python dictionaries (the parsed JSON objects) and the text parts before, between and after the jsons.
        Returns an empty list if no valid JSON is found.
    """
    json_objects = []
    text_parts = []
    try:
        matches =   list(re.finditer(r"\{(?:[^{}]|{[^{}]*})*\}", text)) # Use finditer for indices
        if not matches:
            return [], [text.strip()]  # No JSON found

        last_end = 0
        for match in matches:
            json_string = match.group(0)
            try:
                data = json.loads(json_string)
                json_objects.append(data)
                text_parts.append(text[last_end:match.start()].strip())
                last_end = match.end()
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON
        text_parts.append(text[last_end:].strip()) #add the last part of the text

        return json_objects, text_parts

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], [text.strip()]

# Example usage:
#test_string = 'I think I need to calculate'+ agrep + 'that is the one' +agrep
'''
test_string =  "{\"tool\": \"mytool1\", \"params\": {\"a\": 1} } hiall {\"tool\": \"mytool2\", \"params\": {\"b\": 2} }hiall{\"tool\": \"mytool3\", \"params\": {\"c\": 3} }"

json_data_list, text_parts = extract_all_json(test_string)
print(f"Input: '{test_string}'")
if json_data_list:
    print(f"Extracted JSON objects:")
    for json_data in json_data_list:
        print(json_data)
    print(f"Text parts:")
    print(text_parts)
else:
    print("No valid JSON found.")
    print(f"Original text: '{text_parts[0]}'")
print("-" * 20)
'''

# %%

# %%
import inspect
import os
import importlib

def is_tool_function(obj):
    """
    Checks if a given object is a function and has the "llm tool"as one word tag in its docstring.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a tool function, False otherwise.
    """
    if inspect.isfunction(obj):
        docstring = inspect.getdoc(obj)
        if docstring and "llmtool" in docstring.lower():
            return True
    return False

def list_llm_tools(tools_dir="toolsfn", modules=None):
    """
    Lists all functions defined in the current scope and in modules within a specified directory
    that are intended to be used as tools.

    Args:
        tools_dir (str, optional): The name of the directory containing tool modules. Defaults to "toolsfn".

    Returns:
        list: A list of function objects that are marked as tools.
    """
    tools = []
    if modules is None:
        # Try to get the calling module, with fallback to __main__
        try:
            frame = inspect.currentframe().f_back
            if frame:
                calling_module = inspect.getmodule(frame)
                if calling_module:
                    modules = [calling_module]
                else:
                    import __main__  # Fallback: if no module, assume __main__
                    modules = [__main__]
            else:
                import __main__
                modules = [__main__]
        except Exception as e: #if everything fails, return empty list
            print(f"Error getting calling module: {e}")


    for module in modules:
        for name, obj in inspect.getmembers(module):
            if is_tool_function(obj):
                tools.append(obj)
    toolnames = []
    for tool in tools:
        toolnames.append(tool.__name__)
    # Inspect modules within the specified directory
    if os.path.exists(tools_dir) and os.path.isdir(tools_dir):
        for filename in os.listdir(tools_dir):
            if filename.endswith(".py"):
                module_name = filename[:-3]
                module_path = os.path.join(tools_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for name, obj in inspect.getmembers(module):
                        if is_tool_function(obj) and obj.__name__ not in toolnames:
                            tools.append(obj)
                            toolnames.append(obj.__name__)
                except Exception as e:
                    print(f"Error importing module {module_path}: {e}") #Print error if module cannot be loaded
                    continue #continue with the next file

    return tools


# %%

# %%
class AgentMemory(Agent):
    def __init__(self,*args):
        super().__init__(*args)
        self.memory = ''
        self.responses = []
        self.summarize="You are excellent in summarization, please summarize the below text in less statements. keep all the information you find like names, locations, times, ..etc."
        self.checkmemory="you can check if the infomration is found in a given text or not')"
        self.memoryAgent = agentthis(prompt="",message=self.summarize, modelsel=1, asis=0, tools=[],memory='no') 
    def generate_response(self,prompt,*args):
        if self.memory:
            self.memoryAgent.reset(self.checkmemory)
            found = self.memoryAgent.generate_response('You are given the following text:\n \
                '+'\n'.join(self.responses)+'\n please check if you can find the information regarding the user request: \
                '+ prompt+'\n in this text. if you donot find the requested info then reply: "not found" and do not add any more text\n \
                and if you find the information then reply with this info', 'no')
            if 'not found' not in found.lower():
                print(found)
                return found
        returns = super().generate_response(prompt)
        self.responses.append(self.response)
        self.memoryAgent.reset(self.summarize)
        prompt = self.memory+ '\n' + self.response
        self.memory = self.memoryAgent.generate_response("summarize the following in less number of statements:\n" + prompt, "no")
        #print('selfresponses',self.response)
        print(returns)
        return returns
    
        

# %%

# %%
def agentthis(prompt="",message="", modelsel=1, asis=1, tools = [],memory='no'):
    READER_MODEL_NAME = {}
    READER_MODEL_NAME[1] = "Qwen/Qwen2.5-Coder-7B-Instruct"
    READER_MODEL_NAME[2] = "tiiuae/falcon-7b-instruct"
    READER_MODEL_NAME[3] = 'teknium/OpenHermes-2.5-Mistral-7B'
    READER_MODEL_NAME[4]= 'meta-llama/Llama-3.2-3B-Instruct'
    READER_MODEL_NAME[5] = "mistralai/Mistral-7B-Instruct-v0.3"
    READER_MODEL_NAME[6] = "meta-llama/Llama-3.1-8B"
    READER_MODEL_NAME[7] = "meta-llama/Llama-3.1-8B-Instruct"
    READER_MODEL_NAME[8] = "meta-llama/Meta-Llama-3.1-8b-Instruct"
    READER_MODEL_NAME[9] = "meta-llama/Llama-3.2-1B-Instruct"
    READER_MODEL_NAME[10] = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    READER_MODEL_NAME[11] = "EleutherAI/gpt-neo-2.7B"
    READER_MODEL_NAME[12] = "meta-llama/Llama-3.2-3B"
    READER_MODEL_NAME[13] = "openai/gpt-oss-20b"
    
    if prompt == 'help':
        print("Usage: prompt, system_message, modelsel,asis,tools, memory")
        print("prompt: leave it empty most of the cases, system_message: the role of the model, modelsel: as of below, \n \
                asis: 1 for best quality \n \
                tools: you can define the tools in your code and in the directory:tools_fn under this Agent.py file \n \
                    the tool doc string must contain:'llmtool' \n \
                before defining the agent, run tools = Agent.list_llm_tools(), then add these tools to the agentthis(...) \
                memory == yes/no (keep the summary of the chat)")
        print("it returns: response, agent... ")
        print("you can use the agent.generate_response, and agent.messages, agent.instruct_history to change the memorization")
        print("modelsel can be a number to select the following:")
        print("modelsel = 1: Qwen/Qwen2.5-Coder-7B-Instruct")
        print("modelsel = 2: tiiuae/falcon-7b-instruct")
        print("modelsel = 3: teknium/OpenHermes-2.5-Mistral-7B")
        print("modelsel = 4: meta-llama/Llama-3.2-3B-Instruct")
        print("modelsel = 5: mistralai/Mistral-7B-Instruct-v0.3")
        print("modelsel = 6: meta-llama/Llama-3.1-8B")
        print("modelsel = 7: meta-llama/Llama-3.1-8B-Instruct")
        print("modelsel = 8: meta-llama/Meta-Llama-3.1-8b-Instruct")
        print("modelsel = 9: meta-llama/Llama-3.2-1B-Instruct")
        print("modelsel = 10: mistralai/Mixtral-8x22B-Instruct-v0.1")
        print("modelsel = 11: EleutherAI/gpt-neo-2.7B")
        print("modelsel = 12: meta-llama/Llama-3.2-3B")
        print("modelsel = 13: openai/gpt-oss-20b")
        return
    if memory != "no" :
        agent = AgentMemory(READER_MODEL_NAME[modelsel],'agent1', message, asis, tools)
    else:
        print("start loading -----")
        agent = Agent(READER_MODEL_NAME[modelsel],'agent1', message, asis, tools)
        print("finished loading -----")
    if not prompt:
        prompt = 'Are you ready ?'
    #agent1_response = agent.generate_response(prompt)
    return  agent


# %%
if __name__ == "__main__":
    agent = agentthis("how many continents in the world ? Name them only ", "You are a helpful AI assistant.Your name is Assisto. Maintain context and be concise.\n\n")
    print(agent)
