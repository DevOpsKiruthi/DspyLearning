FROM ubuntu:latest

# Install dependencies
RUN apt update && apt install -y wget gpg curl

# Add Microsoft's GPG key and repository
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/code stable main" | tee /etc/apt/sources.list.d/vscode.list

# Update package list and install VS Code
RUN apt update && apt install -y code

# Install code-server (VS Code in a browser)
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Create a non-root user
RUN useradd -m vscodeuser && chown -R vscodeuser /home/vscodeuser

# Switch to non-root user
USER vscodeuser

# Expose the default code-server port
EXPOSE 8080

# Start code-server on container start
CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none"]
