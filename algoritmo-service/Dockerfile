FROM golang:alpine AS build

WORKDIR /app
COPY . .
RUN go build ./main.go

ENTRYPOINT [ "/app/main" ][]