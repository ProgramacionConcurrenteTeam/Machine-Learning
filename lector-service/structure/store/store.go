package store

type Dataset struct {
	Inputs  [][]interface{} `json:"inputs"`
	Targets []string        `json:"targets"`
}

var Data *Dataset
