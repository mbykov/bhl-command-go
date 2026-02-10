package command

import (
	"testing"
    // "log"
	"fmt"
	// "math"
	// "os"
    // "encoding/json"

	// ort "github.com/yalue/onnxruntime_go"
)


func TestCommand(t *testing.T) {
	// onnxPath := "../Models/all-MiniLM-L6-v2/onnx/model.onnx"
	// libPath := "/home/michael/go/ort/lib/libonnxruntime.so"

	// engine, err := LoadModel(onnxPath, libPath)
	// if err != nil {
	// 	panic(err)
	// }
	// defer ort.DestroyEnvironment()
	// defer engine.Session.Destroy()

	// // Load commands from file
	// jsonData, _ := os.ReadFile("./data/commands.json")
	// var commands []CommandMapping
	// json.Unmarshal(jsonData, &commands)

	// Execute isolated test method
    Command()
	// RunTests(engine, commands)
}


func RunTests(engine *SearchEngine, commands []CommandMapping) {
	testPhrases := []string{
		"пожалуйста стоп запись",
		"начни снимать видео",
		"какая погода в москве",
	}

	fmt.Println("--- Running Intent Recognition Tests ---")
	for _, phrase := range testPhrases {
		result := engine.FindCommand(phrase, commands)
		if result != nil {
			fmt.Printf("Input: [%s]\n  -> Match: %s\n  -> Score: %.4f\n  -> Synonyms: %v\n\n",
				phrase, result.Command, result.Score, result.Synonyms)
		} else {
			fmt.Printf("Input: [%s]\n  -> No command detected (below threshold)\n\n", phrase)
		}
	}
}

func ExampleRunTests() {
    fmt.Printf("_kuku")
}
