package endpoint

import (
	"context"
	"errors"

	"github.com/go-kit/kit/endpoint"
	"github.com/rodrities/lector-service/structure/entity"
	"github.com/rodrities/lector-service/structure/service"
)

func MakeLoadDatasetEndpoint(svc service.DatasetService) endpoint.Endpoint {
	return func(_ context.Context, request interface{}) (interface{}, error) {
		inputs, targets, err := svc.LoadDataset()
		if err != nil {
			return nil, errors.New("Error loading dataset")
		}
		return entity.LoadDatasetResponse{Inputs: inputs, Targets: targets}, nil
	}
}
