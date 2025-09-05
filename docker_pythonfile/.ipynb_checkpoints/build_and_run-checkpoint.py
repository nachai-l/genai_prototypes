import docker
from pathlib import Path

client = docker.from_env()

# Build an image from a local Dockerfile
image, logs = client.images.build(
    path=str(Path(".").resolve()),
    tag="myapp:py-sdk",
    rm=True
)
for log in logs:
    print(log.get("stream",""), end="")

# Run a container and capture output
container = client.containers.run("myapp:py-sdk", detach=True)
logs = container.logs(stream=True)
for line in logs:
    print(line.decode().rstrip())

exit_code = container.wait()["StatusCode"]
print("Exit:", exit_code)
container.remove()
