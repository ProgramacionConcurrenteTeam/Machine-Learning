package service

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/dustin/go-humanize"
	"github.com/rodrities/lector-service/structure/store"
)

// ErrEmpty is returned when an input string is empty.
var ErrEmpty = errors.New("empty string")

type WriteCounter struct {
	Total uint64
}

func (wc *WriteCounter) Write(p []byte) (int, error) {
	n := len(p)
	wc.Total += uint64(n)
	wc.PrintProgress()
	return n, nil
}

func (wc WriteCounter) PrintProgress() {
	// Clear the line by using a character return to go back to the start and remove
	// the remaining characters by filling it with spaces
	fmt.Printf("\r%s", strings.Repeat(" ", 35))

	// Return again and print current status of download
	// We use the humanize package to print the bytes in a meaningful way (e.g. 10 MB)
	fmt.Printf("\rDescargando... %s completado", humanize.Bytes(wc.Total))
}

func DownloadFile(filepath string, url string) error {

	// Create the file, but give it a tmp file extension, this means we won't overwrite a
	// file until it's downloaded, but we'll remove the tmp extension once downloaded.
	out, err := os.Create(filepath + ".tmp")
	if err != nil {
		return err
	}

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		out.Close()
		return err
	}
	defer resp.Body.Close()

	// Create our progress reporter and pass it to be used alongside our writer
	counter := &WriteCounter{}
	if _, err = io.Copy(out, io.TeeReader(resp.Body, counter)); err != nil {
		out.Close()
		return err
	}

	// The progress use the same line so print a new line once it's finished downloading
	fmt.Print("\n")

	// Close the file without defer so it can happen before Rename()
	out.Close()

	if err = os.Rename(filepath+".tmp", filepath); err != nil {
		return err
	}
	return nil
}

// DatasetService provides operations on strings.
type DatasetService interface {
	LoadDataset() ([][]interface{}, []string, error)
}

// datasetService is a concrete implementation of DatasetService
type datasetService struct{}

func NewDatasetService() DatasetService {
	return &datasetService{}
}

func (datasetService) LoadDataset() ([][]interface{}, []string, error) {
	// Lectura del dataset
	if _, err := os.Stat("TB_F00_SICOVID.csv"); err == nil {
		fmt.Println("Abriendo dataset obtenido de la web")

	} else if errors.Is(err, os.ErrNotExist) {

		DownloadFile("TB_F00_SICOVID.csv", "https://raw.githubusercontent.com/ProgramacionConcurrenteTeam/Machine-Learning/main/TB_F00_SICOVID.csv")

	}
	f, _ := os.Open("TB_F00_SICOVID.csv")
	N_ROWS := 1000
	//f, _ := os.Open(path) //"TB_F00_SICOVID.csv"
	//f, _ := os.Open("TB_F00_SICOVID.csv") //"TB_F00_SICOVID.csv"

	defer f.Close()
	// Leer todo lo demás del dataset
	content, _ := ioutil.ReadAll(f)
	s_content := string(content)
	lines := strings.Split(s_content, "\n")

	// Declarar inputs y target
	inputs := make([][]interface{}, 0)
	targets := make([]string, 0)
	for i, line := range lines {
		// primera linea
		line = strings.TrimRight(line, "\r\n")
		// si esta vacia, ignorar esta iteracion
		if len(line) == 0 {
			continue
		}
		// ignorar header
		if i == 0 {
			continue
		}
		// crear arreglo
		tup := strings.Split(line, ",")
		// arreglo con indices de 4-18 los cuales seran los inputs
		pattern := tup[0:2]
		// arreglo con el target que se encuentra en el indice 2
		target := tup[2]
		// arreglo X que obtendra los inputs convertidos de dato string a float64
		X := make([]interface{}, 0)
		for _, x := range pattern {
			f_x, _ := strconv.ParseFloat(x, 64)
			X = append(X, f_x)
		}
		// se alamcena en el arreglo inputs
		inputs = append(inputs, X)
		// se alamacena en el arreglo target
		targets = append(targets, target)
		// si llegamos al limite de filas, romper el bucle
		if i == N_ROWS {
			break
		}
	}

	d := store.Dataset{Inputs: inputs, Targets: targets}
	store.Data = &d
	return inputs, targets, nil
}
