import modal

app = modal.App()

@app.function(gpu="A10G")  # defining a Modal Function with a GPU
def check_gpus():
    import subprocess

    print("here's my gpu:")
    try:
        subprocess.run(["nvidia-smi", "--list-gpus"], check=True)
    except Exception:
        print("no gpu found :(")

@app.local_entrypoint()  # defining a CLI entrypoint
def main():
    print("hello from the .local playground!")
    check_gpus.local()

    print("let's try this .remote-ly on Modal...")
    check_gpus.remote()