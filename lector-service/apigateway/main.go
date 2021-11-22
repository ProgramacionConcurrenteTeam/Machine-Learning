package main

import (
	"net/http"

	"github.com/rodrities/lector-service/structure/endpoint"
	"github.com/rodrities/lector-service/structure/handler"
	"github.com/rodrities/lector-service/structure/service"
)

func main() {

	svc := service.NewDatasetService()
	loadDatasetEndpoint := endpoint.MakeLoadDatasetEndpoint(svc)
	handler.NewHttpHandler(loadDatasetEndpoint)
	http.ListenAndServe(":8081", nil)

}
