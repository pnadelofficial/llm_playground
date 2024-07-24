from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Javascript
import os
from threading import Thread
import json

class ModelSelector:
    def __init__(self):
        pass

    def init_dropdown(self):
        self.dd = widgets.Dropdown(
            options=[m.split('--')[1]+'/'+m.split('--')[2] for m in os.listdir('/cluster/tufts/tuftsai/models') if m.startswith('models--')],
            description='Model:',
            disabled=False,
        )
        
    def __call__(self):
        self.init_dropdown()
        display(self.dd)

class LLM:
    def __init__(self, model_path):
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.llm_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, cache_dir='/cluster/tufts/tuftsai/models')
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_path, quantization_config=self.bnb_config, device_map="auto", cache_dir='/cluster/tufts/tuftsai/models')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.streamer = TextIteratorStreamer(self.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chat = []
            
    def ta(self):
        ta = widgets.Textarea(
            value='',
            placeholder='Type something',
            description='User:',
            disabled=False,
            layout=widgets.Layout(height="90%", width="auto")
        )
        return ta

    def button_output(self):
        self.button = widgets.Button(description="Submit message")
        self.button.on_click(self.llm_call)
        self.output = widgets.Output()
        self.ta = self.ta()

    def display_single_message(self, message):
        display(HTML(f"""
                    <div style="border: 1px solid black; padding: 10px;">
                    <p><b>{message['role'].title()}: </b>{message['content']}</p>
                    <div>
                """.strip())) 
    
    def llm_call(self, b):
        with self.output:
            # Get the user input before clearing the input area
            user_input = self.ta.value

            # Clear the input area after capturing the input
            self.ta.value = ''

            # Process the new user message
            if user_input.strip():  # Only process non-empty messages
                new_user_message = {'role': 'user', 'content': user_input}
                self.chat.append(new_user_message)

                # Create a unique ID for this assistant message
                assistant_message_id = f"assistant-message-{len(self.chat)}"

                # Clear previous output and display all messages including the new user message
                clear_output(wait=True)
                for message in self.chat:
                    self.display_single_message(message)

                # Prepare for the new assistant message
                display(HTML(f"""
                    <div id="{assistant_message_id}" style="border: 1px solid black; padding: 10px;">
                        <p id="streamOutput-{assistant_message_id}">
                            <b>Assistant: </b>
                        </p>
                    </div>
                """))

                display(Javascript(f"""
                    (function() {{
                        var streamOutput = document.getElementById("streamOutput-{assistant_message_id}");
                        var lastText = "";
                        window["appendText_{assistant_message_id}"] = function(text) {{
                            if (text !== lastText) {{
                                streamOutput.innerHTML += text;
                                lastText = text;
                            }}
                        }};
                    }})();
                """))

                tokenized_chat = self.tokenizer.apply_chat_template(self.chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                generation_kwargs = dict(inputs=tokenized_chat.to(self.device), streamer=self.streamer, max_new_tokens=4000, pad_token_id=self.tokenizer.eos_token_id)
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                generated_text = ''
                for new_text in self.streamer:
                    escaped_text = json.dumps(new_text.replace('\n', '<br>').replace('"', '\\"'))
                    display(Javascript(f"window['appendText_{assistant_message_id}']({escaped_text});"))
                    generated_text += new_text

                # After generation is complete, add the assistant's message to the chat history
                new_assistant_message = {'role': 'assistant', 'content': generated_text}
                self.chat.append(new_assistant_message)

                # Clear output and redisplay all messages to ensure correct order
                clear_output(wait=True)
                for message in self.chat:
                    self.display_single_message(message)

                # Clean up the generated text
                o = generated_text.replace('\\"', '"') #.replace('\n', '<br>')
            else:
                print("Please enter a message before submitting.")
            
    def display(self):
        display(self.output)
        display(self.ta)
        display(self.button)

    def __call__(self):
        self.button_output()
        self.display()

        
#                     <html>
#                         <body>
#                             <script>
#                                 function appendText() {{
#                                     var paragraph = document.getElementById("streamOutput");
#                                     var newText = {escaped_text};
#                                     paragraph.innerHTML += newText;
#                                 }}
#                                 appendText();
#                             </script>
#                         </body>
#                     </html>