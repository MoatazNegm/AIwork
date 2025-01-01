#!/usr/bin/env python
# coding: utf-8
# %%

# %%

# %%

# %%
# #!pip install langgraph
# #!pip3  install torch torchvision torchaudio transformers
# #!pip3 install packaging ninja
# #!pip3 install accelerate
# #!pip3 install protobuf
# #!pip3 install sentencepiece
# #!pip3 install bitsandbytes
# #!pip3 install scipy

from transformers import pipeline, Tool
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
import random, json
import inspect
import json, re
from typing import Dict, Any, Optional, Callable, List

class Agent:
    def __init__(self, model_name='Qwen/Qwen2.5-Coder-7B-Instruct',
                agent_name='dummy_model', message='', asis=1, tools=[]):
        # Load Qwen model and tokenizer            
        max_length=8500  # Total tokens (input + output)
        max_new_tokens=500  # Limit output tokens
        self.tools = tools
        self.schema_tools = []
        self.pipelineTools = []
        self.instruct_history = 10
        self.instruct_history_counter = 0
    

        self.initTools(self.tools)
        
        if asis == 1:
            save_directory = '../'+model_name.replace('/','_')+'_saved_quality'
            torchfloat = torch.bfloat16
        else:
            save_directory = '../'+model_name.replace('/','_')+'_saved_response'
            torchfloat = torch.float16
        
        try:
            
            print('Trying to load the mode:',save_directory,'from local repo')
            self.model = AutoModelForCausalLM.from_pretrained(save_directory)
            self.tokenizer = AutoTokenizer.from_pretrained(save_directory)
            #self.model_pipeline = pipeline(
            #    "text-generation", 
            #    model=save_directory,
            #    model_kwargs={
            #        #"quantization_config": bnb_config,
            #        "torch_dtype": torchfloat,
            #        "device_map": "cuda",
            #       
            #    },
            #    max_length=max_length,
            #    max_new_tokens=max_new_tokens,
            #    truncation=True,
            #    temperature = 0.1,
            #    top_k=1,
            #    tools= self.pipelineTools
            #)
        except:
            print('The model:',model_name,'is not found locally, downloading it')
            if asis == 1:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Enable 4-bit quantization with careful settings
                    bnb_4bit_quant_type="nf4",  # Normalized float 4 quantization
                    bnb_4bit_compute_dtype=torch.bfloat16,  # More stable computation dtype
                    bnb_4bit_use_double_quant=True,  # Enable double quantization for better compression
                )
                # Convert BitsAndBytesConfig to a dictionary for saving
                quantization_config_dict = {
                    "load_in_4bit": bnb_config.load_in_4bit,
                    "bnb_4bit_quant_type": bnb_config.bnb_4bit_quant_type,
                    "bnb_4bit_compute_dtype": str(bnb_config.bnb_4bit_compute_dtype),
                    "bnb_4bit_use_double_quant": bnb_config.bnb_4bit_use_double_quant
                }
            else:
                bnb_config = BitsAndBytesConfig(
                    torch_dtype="auto",
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quantw_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,  # Changed from bfloat16 to float16
                    bnb_4bit_quant_storage=torch.uint8,    # Added for storage optimization
                    use_nested_quant=True,                 # Added for nested quantization
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, token="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"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
            #self.model_pipeline = pipeline(
            #    "text-generation", 
            #    model=model_name,
            #    model_kwargs={
            #        #"quantization_config": bnb_config,
            #        "torch_dtype": torchfloat,
            #        "device_map": "cuda",
            #        
             ##   },
            #    max_length=max_length,
            #    max_new_tokens=max_new_tokens,
            #    truncation=True,
            #    token="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",
            #    temperature = 0.1,
            #    top_k=1,
            #    tools= self.pipelineTools
            #)
            print("Saving the model:",model_name," locally")
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            #self.model_pipeline.model.save_pretrained(save_directory)
            #self.model_pipeline.tokenizer.save_pretrained(save_directory)
            
            
        self.agent_name = agent_name
        self.model_name = model_name
        if not message:
            message = "You are a helpful AI assistant. Maintain context and be concise.\n\n"
        self.message = message
        if 'nstruct' in model_name:
                self.messages = [dict({"role": "system", "content": self.message})]
        else:
            self.messages = ""
               
    def get_tool_schema(self,func: Callable) -> dict:
        """
        Generate a JSON schema for a tool function.

        Args:
            func (Callable): The function to generate a schema for.

        Returns:
            dict: A JSON schema representing the function's parameters.
        """
        import inspect
        signature = inspect.signature(func)
        parameters = {}

        for name, param in signature.parameters.items():
            parameters[name] = {
                "type": "string",  # Assume string type for simplicity
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
        
    def initTools(self,tools=0):
        self.schema_tools = []
        self.pipelineTools = []
        if  tools != 0:
            self.tools = tools
        for tool in self.tools:
            self.schema_tools.append(self.get_tool_schema(tool))
            self.pipelineTools.append(Tool(
                                        name=tool.__name__,
                                        func=tool,
                                        description=inspect.getdoc(tool)
                                    ))

              
    def create_system_prompt(self,prompt, extras=[]):
            if 'nstruct'in self.model_name:
                result = self.instruct_create_system_prompt(prompt)
                
                return result
            else:
                return self.llm_create_system_prompt(prompt)

    def instruct_create_system_prompt(self,prompt, extras=[]):  ### for instruct models
        
        self.messages.append(dict({
            "role": "user", 
            "content": prompt
        }))
        #for msg in self.messages[-self.instruct_history:]:  # Limit context to last 5 messages
        #    prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        #prompt += "Assistant:"
        
        self.messages =  self.messages[-self.instruct_history:]
        if self.message not in str(self.messages[0]['content']):
            init_message = [dict({"role": "system", "content": self.message})]
            self.messages = init_message + self.messages
        return self.messages
            
    def llm_create_system_prompt(self,prompt):
        return self.system_messages+'\n'+prompt
    
        
    def get_tool_schema(self,func: Callable) -> dict:
        """
        Generate a JSON schema for a tool function.

        Args:
            func (Callable): The function to generate a schema for.

        Returns:
            dict: A JSON schema representing the function's parameters.
        """
        import inspect
        signature = inspect.signature(func)
        parameters = {}

        for name, param in signature.parameters.items():
            parameters[name] = {
                "type": "string",  # Assume string type for simplicity
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
    def generate_response(self,prompt):
        if 'nstruct' in self.model_name:
            return self.instruct_generate_response(prompt)
        else:
            return self.llm_generate_response(prompt)
        
    def instruct_generate_response(self,prompt):  #### given we are using instruct model and it is  tools compatible (if tools are stated)
        
        # Generate response
        messages =  self.create_system_prompt(prompt)
        
        #response = self.model_pipeline(messages)[0]['generated_text']
        #self.messages.append({
        #    "role": "assistant", 
        #    "content": response.split('ssistant:')[-1].split("User")[0],
            
        #})
        
        
        #return response.split('ssistant:')[-1].split("User")[0]

        text = self.tokenizer.apply_chat_template(
            messages,
            tools= self.tools,
            tokenize=False,
            add_generation_prompt=True,
            #return_tensors="pt"
        )


        # Generate response
       
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=True).to(self.model.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            #inputs,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=384,
            temperature = 0.1,
            #pad_token_id=self.model.config.eos_token_id
        )

        # Decode response
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.messages.append({
            "role": "assistant", 
            "content": response.split('ssistant\n\n')[-1].split("User\n\n")[0].split("user\n\n")[0],
            
        })
        cleaned_response = response.split('ssistant\n\n')[-1].split("User\n\n")[0].split("user\n\n")[0]
        jsons, _ = json_data_list, text_parts = self.extract_all_json(cleaned_response)
        if len(jsons) > 0:
            for json_data in jsons:
                print('tool_call',json_data)
                #json_result = globals().get(json_data['name'])(**json_data['parameters'])
                json_result = self.tools[1](**json_data['parameters'])
                print('tool_result',json_result)
                #tool_call = {"name": "list_files", "arguments": {"location": "Paris, France"}}
                #messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
        return cleaned_response
    
    def extract_all_json(self,text):
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
    
    def reset(self,message=''):
        if len(message) == 0:
            self.messages = [self.messages[0]]
        else:
            self.messages = [{"role":"system", "content":message}]
        return
    
    def llm_generate_response(self, prompt): # if the model is not instruct
        # Prepare input
        
        messages = self.create_system_prompt(prompt)
        text = messages
        print('llm result: ')
        response = self.model_pipeline(prompt)[0]['generated_text']
        return response[len(prompt):].strip()
        # Generate response
        inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            #input_ids=inputs["input_ids"],
            **inputs,
            max_new_tokens=100, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.model.config.eos_token_id
        )
        
        # Decode response
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response[len(text):].strip()
    



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
def agentthis(prompt="",message="", modelsel=1, asis=1, tools = []):
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
    if prompt == 'help':
        print("Usage: prompt, system_message, modelsel,asis")
        print("prompt: leave it empty most of the cases, system_message: the role of the model, modelsel: as of below, asis: 1 for best quality")
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
        return
    
    agent = Agent(READER_MODEL_NAME[modelsel],'agent1', message, asis, tools)
    if not prompt:
        prompt = 'Are you ready ?'
    #agent1_response = agent.generate_response(prompt)
    return  agent


# %%
if __name__ == "__main__":
    agent = agentthis("how many continents in the world ? Name them only ", "You are a helpful AI assistant.Your name is Assisto. Maintain context and be concise.\n\n")
    print(agent)
