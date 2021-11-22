FROM golang:alpine AS build

WORKDIR /app
COPY . .
RUN go build ./apigateway/main.go

ENTRYPOINT [ "/app/main" ][]