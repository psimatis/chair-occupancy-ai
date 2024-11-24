FROM node:18-alpine
WORKDIR /app
COPY . .
RUN npm install http-server
EXPOSE 80
CMD ["npx", "http-server", "-p", "80"]
