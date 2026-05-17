!pip install -q gradio transformers accelerate

import torch
import gradio as gr
import warnings

from google.colab import drive
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/it_helpdesk_llama_lora_merged_model"

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Modelo cargado correctamente.")

system_message = (
    "You are an IT Helpdesk support chatbot. "
    "Always answer in English. "
    "Your answer must include a diagnosis and a clear solution. "
    "Use this exact format: Diagnosis: ... Solution: ..."
)

def ask_chatbot(user_problem):
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_problem
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return answer.strip()

test_questions = [
    "I cannot connect to the WiFi",
    "My computer is very slow",
    "I forgot my password",
    "The printer does not print",
    "I cannot open Word",
    "I am not receiving emails",
    "My camera does not work in Zoom",
    "I think my computer has a virus",
    "Windows does not boot",
    "I cannot access Moodle",
    "I get error 500",
    "My USB drive does not appear on the computer",
    "The microphone does not work in Teams",
    "I opened a suspicious link in an email"
]

def load_initial_chat():
    history = []

    welcome_message = (
        "Welcome! I am your IT Helpdesk Chatbot.\n\n"
        "I will first answer some automatic test questions. "
        "After that, you can write your own technical problem below."
    )

    history.append({
        "role": "assistant",
        "content": welcome_message
    })

    for question in test_questions:
        answer = ask_chatbot(question)

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

    return history

def respond(message, history):
    if history is None:
        history = []

    message = message.strip()

    if message == "":
        history.append({
            "role": "assistant",
            "content": "Please describe your technical problem."
        })
        return history, ""

    history.append({
        "role": "user",
        "content": message
    })

    answer = ask_chatbot(message)

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history, ""

def clear_chat():
    return [
        {
            "role": "assistant",
            "content": "Chat cleared. Write a technical problem and I will help you diagnose it."
        }
    ]

def reload_examples():
    return load_initial_chat()

def example_wifi(history):
    return respond("I cannot connect to the WiFi", history)

def example_password(history):
    return respond("I forgot my password", history)

def example_printer(history):
    return respond("The printer does not print", history)

def example_virus(history):
    return respond("I opened a suspicious link in an email", history)

custom_css = """
body {
    background: #eef3f9;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding-top: 10px !important;
}

#title {
    text-align: center;
    color: #0f172a;
    font-size: 44px;
    font-weight: 900;
    margin-bottom: 8px;
}

#subtitle {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin-bottom: 24px;
}

#chatbot {
    border-radius: 24px;
    border: 2px solid #bfdbfe;
    background: white !important;
    box-shadow: 0px 8px 26px rgba(15, 23, 42, 0.12);
}

#status-box {
    background: #ffffff;
    border: 2px solid #dbeafe;
    border-radius: 22px;
    padding: 20px;
    color: #0f172a;
    box-shadow: 0px 6px 20px rgba(15, 23, 42, 0.08);
}

#tips-box {
    background: #ffffff;
    border: 2px solid #dbeafe;
    border-radius: 22px;
    padding: 20px;
    color: #0f172a;
    box-shadow: 0px 6px 20px rgba(15, 23, 42, 0.08);
}

#status-box h3, #tips-box h3 {
    margin-top: 0;
    color: #1d4ed8;
    font-size: 25px;
    font-weight: 800;
}

#status-box p, #tips-box li {
    color: #1e293b !important;
    font-size: 16px;
    line-height: 1.5;
}

button {
    border-radius: 13px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

textarea {
    border-radius: 16px !important;
}

label {
    color: #0f172a !important;
    font-weight: 700 !important;
}

footer {
    visibility: hidden;
}
"""
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    ),
    css=custom_css,
    title="IT Helpdesk Chatbot"
) as demo:

    gr.HTML("""
    <div id="title">IT Helpdesk Chatbot</div>
    <div id="subtitle">
        Fine-tuned chatbot using Llama + LoRA for technical support diagnosis and solutions.
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=620,
                elem_id="chatbot"
            )

            msg = gr.Textbox(
                label="Describe your technical problem",
                placeholder="Example: I cannot connect to the WiFi...",
                lines=2
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                reload_btn = gr.Button("Run automatic tests")
                clear_btn = gr.Button("Clear chat")

            with gr.Row():
                wifi_btn = gr.Button("Example: WiFi")
                password_btn = gr.Button("Example: Password")
                printer_btn = gr.Button("Example: Printer")
                virus_btn = gr.Button("Example: Phishing")

        with gr.Column(scale=1):

            gr.HTML("""
            <div id="status-box">
                <h3>Model Status</h3>
                <p><b>Model:</b> Llama 3.2 1B Instruct</p>
                <p><b>Adaptation:</b> LoRA fine-tuning</p>
                <p><b>Execution:</b> Google Colab</p>
                <p><b>Device:</b> GPU / CUDA</p>
                <p><b>Output:</b> Diagnosis + Solution</p>
            </div>
            """)

            gr.HTML("""
            <div id="tips-box">
                <h3>Try asking:</h3>
                <ul>
                    <li>I forgot my password.</li>
                    <li>The printer does not print.</li>
                    <li>My computer is very slow.</li>
                    <li>I cannot access Moodle.</li>
                    <li>Windows does not boot.</li>
                    <li>I opened a suspicious email link.</li>
                </ul>
            </div>
            """)
    demo.load(
        fn=load_initial_chat,
        inputs=None,
        outputs=chatbot
    )

    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    reload_btn.click(
        fn=reload_examples,
        inputs=None,
        outputs=chatbot
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot
    )

    wifi_btn.click(
        fn=example_wifi,
        inputs=[chatbot],
        outputs=[chatbot, msg]
    )

    password_btn.click(
        fn=example_password,
        inputs=[chatbot],
        outputs=[chatbot, msg]
    )

    printer_btn.click(
        fn=example_printer,
        inputs=[chatbot],
        outputs=[chatbot, msg]
    )

    virus_btn.click(
        fn=example_virus,
        inputs=[chatbot],
        outputs=[chatbot, msg]
    )

demo.queue()

demo.launch(
    share=True,
    debug=False
)