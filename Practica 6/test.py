import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "unsloth/Llama-3.2-1B-Instruct"
lora_path = "./lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(lora_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if device == "cuda" else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    device_map=None
)

base_model = base_model.to(device)

model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    is_trainable=False
)

model = model.to(device)
model.eval()

def ask(prompt):
    full_prompt = f"""You are a Linux command generator. Output only the command, no explanation.

User: {prompt}
Assistant:"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Assistant:")[-1].strip().split("\n")[0]

tests = [
    # Files and directories
    "List all files in the current directory",
    "List all files in the current directory with details",
    "Show hidden files in the current folder",
    "Create a directory called backups",
    "Remove an empty directory called old_folder",
    "Delete a file called notes.txt",
    "Copy file.txt to the backup directory",
    "Move report.pdf to the Documents folder",
    "Rename old.txt to new.txt",

    # cat/head/tail
    "Show the content of file.txt",
    "Show the first 5 lines of a file called log.txt",
    "Show the first 20 lines of notes.txt",
    "Show the last 10 lines of system.log",
    "Follow a log file in real time",

    # find
    "Find all files ending in .log in the current folder",
    "Find all .py files recursively",
    "Search for python files in current directory recursively",
    "Find all text files in the current directory and subdirectories",
    "Find files larger than 100MB",
    "Find files modified in the last 24 hours",
    "Find a file named config.json",

    # grep
    "Search for the word error inside log.txt",
    "Search recursively for the word TODO in the current directory",
    "Show lines containing warning in system.log with line numbers",
    "Search for text ignoring case in a file",

    # disk and memory
    "Show overall disk space usage",
    "Display disk usage summary",
    "Show folder size in human readable format",
    "Check memory usage",
    "Show RAM usage in human readable format",

    # processes
    "List all running processes",
    "Find the process ID of nginx",
    "Kill a process with PID 1234",
    "Kill a process named firefox",
    "Show CPU usage",

    # network
    "Check my IP address",
    "Show my public IP address",
    "Ping google.com",
    "Show active network connections",
    "Show listening ports",
    "Check which process is using port 8080",

    # permissions
    "Change permissions of script.sh to executable",
    "Set permissions of file.txt to 755",
    "Change owner of file.txt to user and group",
    "Create a symbolic link to file.txt",

    # compression
    "Create a tar.gz archive of the folder project",
    "Extract archive.tar.gz",
    "Create a zip file from folder docs",
    "Unzip archive.zip",

    # system
    "Show current date and time",
    "Show system uptime",
    "Show kernel version",
    "Show operating system version",
    "Show current user",
    "Show current shell",

    # packages/services
    "Install a package called git",
    "Remove package nginx",
    "Update package list",
    "Start nginx service",
    "Stop nginx service",
    "Restart nginx service",
    "Check nginx service status",
    "Make a push on git",
    "How do I find all files ending in .log in the current folder?"
]

for t in tests:
    print("INPUT:", t)
    print("OUTPUT:", ask(t))
    print("-" * 50)