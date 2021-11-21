package main

import (
	"log"

	"./logisticregression"
)

func main() {
	if err := logisticregression.Run(); err != nil {
		log.Fatal(err)
	}
}
