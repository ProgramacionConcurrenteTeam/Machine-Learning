package entity

type LoadDatasetResponse struct {
	Inputs  [][]interface{} `json:"inputs"`
	Targets []string        `json:"targets"`
}
type LoadDatasetRequest struct {
	N_ROWS int `json:"n_rows"`
}
