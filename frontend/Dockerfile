# Use a Node.js base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies, including http-server
RUN npm install http-server

# Expose the default HTTP port
EXPOSE 80

# Start the frontend server
CMD ["npx", "http-server", "-p", "80"]
