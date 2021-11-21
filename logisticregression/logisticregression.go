package logisticregression

import (
	"fmt"
	"io/ioutil"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
)

// Para evaluar el rendimiento de nuestro modelo entrenado en el
// conjunto de pruebas. Para ello, crearemos la denominada
// Matriz de Confusión con la siguiente estructura de datos

type ConfusionMatrix struct {
	positive int //El numero de ejemplos positivos

	negative int //El numero de ejemplos negativos

	truePositive int //El numero de ejemplos positivos que se
	//predijeron de manera correcta

	trueNegative int //El numero de ejemplos negativos que se
	//predijeron de manera correcta

	falsePositive int //El numero de ejemplos positivos que se
	//predijeron de manera incorrecta

	falseNegative int //El numero de ejemplos negativos que se
	//predijeron de manera incorrecta

	recall    float64
	precision float64
	accuracy  float64 //medidia para la precision del modelo
	//este esta definido como (truePositive + trueNegative) / (positive + negative)
}

func (cm ConfusionMatrix) String() string {
	return fmt.Sprintf("\tPositives: %d\n\tNegatives: %d\n\tTrue Positives: %d\n\tTrue Negatives: %d\n\tFalse Positives: %d\n\tFalse Negatives: %d\n\n\tRecall: %.2f\n\tPrecision: %.2f\n\tAccuracy: %.2f\n",
		cm.positive, cm.negative, cm.truePositive, cm.trueNegative, cm.falsePositive, cm.falseNegative, cm.recall, cm.precision, cm.accuracy)
}

func Run() error {
	fmt.Println("Running Logistic Regression...")
	// Cargando la data
	//Cambiar al endpoint
	xTrain, yTrain, err := base.LoadDataFromCSV("./data/studentsTrain.csv")
	if err != nil {
		return err
	}
	xData, yData, err := base.LoadDataFromCSV("./data/studentsTest.csv")
	if err != nil {
		return err
	}

	var maxAccuracy float64
	var maxAccuracyCM *ConfusionMatrix
	var maxAccuracyDb float64
	var maxAccuracyModel *linear.Logistic

	for iter := 100; iter < 3000; iter += 500 {
		for db := 0.05; db < 1.0; db += 0.01 {
			cm, model, err := tryValues(0.0001, 0.0, iter, db, xTrain, xData, yTrain, yData)
			if err != nil {
				return err
			}
			if cm.accuracy > maxAccuracy {
				maxAccuracy = cm.accuracy
				maxAccuracyCM = cm
				maxAccuracyDb = db
				maxAccuracyModel = model

			}
		}
	}

	fmt.Printf("Maxima Precicion: %.2f\n\n", maxAccuracy)
	fmt.Printf("Con el modelo: %s\n\n", maxAccuracyModel)
	fmt.Printf("Con la Matriz de confusion:\n%s\n\n", maxAccuracyCM)
	fmt.Printf("con límite de decisión: %.2f\n", maxAccuracyDb)

	return nil
}

func tryValues(learningRate float64, regularization float64, iterations int, decisionBoundary float64, xTrain, xData [][]float64, yTrain, yData []float64) (*ConfusionMatrix, *linear.Logistic, error) {
	//Recopilando todos los valores positivos y negativos del data set

	cm := ConfusionMatrix{}
	for _, y := range yData {
		if y == 1.0 {
			cm.positive++
		}
		if y == 0.0 {
			cm.negative++
		}
	}

	//entrenando el modelo
	model := linear.NewLogistic(base.BatchGA, learningRate, regularization, iterations, xTrain, yTrain)
	model.Output = ioutil.Discard
	err := model.Learn()
	if err != nil {
		return nil, nil, err
	}

	// Se itira sonre el dataset y se predicen los resultados por cada putno
	//del dato y se registra en la matriz de confusion
	for i := range xData {
		prediction, err := model.Predict(xData[i])
		if err != nil {
			return nil, nil, err
		}
		y := int(yData[i])
		positive := prediction[0] >= decisionBoundary

		if y == 1 && positive {
			cm.truePositive++
		}
		if y == 1 && !positive {
			cm.falseNegative++
		}
		if y == 0 && positive {
			cm.falsePositive++
		}
		if y == 0 && !positive {
			cm.trueNegative++
		}
	}

	// Calculando las metricas
	cm.recall = float64(cm.truePositive) / float64(cm.positive)
	cm.precision = float64(cm.truePositive) / (float64(cm.truePositive) + float64(cm.falsePositive))
	cm.accuracy = float64(float64(cm.truePositive)+float64(cm.trueNegative)) / float64(float64(cm.positive)+float64(cm.negative))
	return &cm, model, nil
}
