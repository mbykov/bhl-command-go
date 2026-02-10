package command

import (
    // "log"
	// "fmt"
	"math"
	"os"
    "encoding/json"

	ort "github.com/yalue/onnxruntime_go"
    "github.com/sugarme/tokenizer/pretrained"
    "github.com/sugarme/tokenizer"
)

// CommandMapping represents the full command object from JSON
type CommandMapping struct {
	Command  string   `json:"command"`
	Synonyms []string `json:"synonyms"`
	Score    float32  `json:"score,omitempty"` // Added to hold the result score
}

type SearchEngine struct {
	Tokenizer *tokenizer.Tokenizer
	Session   *ort.DynamicAdvancedSession
}

// 1. Extraction: Model Loading Logic
func LoadModel(onnxPath, libPath string) (*SearchEngine, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}

	tk := pretrained.BertBaseUncased()
	session, err := ort.NewDynamicAdvancedSession(onnxPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return &SearchEngine{Tokenizer: tk, Session: session}, nil
}

// 2. Extraction: Search Logic (Returns full object + score)
func (se *SearchEngine) FindCommand(inputPhrase string, commands []CommandMapping) *CommandMapping {
	baseVec := se.getEmbedding(inputPhrase)
	var bestMatch *CommandMapping
	var maxSimilarity float32 = -1.0

	for _, cmdEntry := range commands {
		for _, synonym := range cmdEntry.Synonyms {
			synVec := se.getEmbedding(synonym)
			score := cosineSimilarity(baseVec, synVec)

			if score > maxSimilarity {
				maxSimilarity = score
				// Create a copy to return with the score
				bestMatch = &CommandMapping{
					Command:  cmdEntry.Command,
					Synonyms: cmdEntry.Synonyms,
					Score:    score,
				}
			}
		}
	}

	if maxSimilarity > 0.83 {
		return bestMatch
	}
	return nil
}

// // 3. Extraction: Testing Logic
// func RunTests(engine *SearchEngine, commands []CommandMapping) {
// 	testPhrases := []string{
// 		"пожалуйста стоп запись",
// 		"начни снимать видео",
// 		"какая погода в москве",
// 	}

// 	fmt.Println("--- Running Intent Recognition Tests ---")
// 	for _, phrase := range testPhrases {
// 		result := engine.FindCommand(phrase, commands)
// 		if result != nil {
// 			fmt.Printf("Input: [%s]\n  -> Match: %s\n  -> Score: %.4f\n  -> Synonyms: %v\n\n",
// 				phrase, result.Command, result.Score, result.Synonyms)
// 		} else {
// 			fmt.Printf("Input: [%s]\n  -> No command detected (below threshold)\n\n", phrase)
// 		}
// 	}
// }

// func main() {
func Command() {
	onnxPath := "../Models/all-MiniLM-L6-v2/onnx/model.onnx"
	libPath := "/home/michael/go/ort/lib/libonnxruntime.so"

	engine, err := LoadModel(onnxPath, libPath)
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()
	defer engine.Session.Destroy()

	// Load commands from file
	jsonData, _ := os.ReadFile("./data/commands.json")
	var commands []CommandMapping
	json.Unmarshal(jsonData, &commands)

    // return engine, commands
	// Execute isolated test method
	RunTests(engine, commands)
}

// --- Internal Helpers ---

func (se *SearchEngine) getEmbedding(text string) []float32 {
	en, _ := se.Tokenizer.EncodeSingle(text)
	seqLen := int64(len(en.Ids))
	shape := ort.NewShape(1, seqLen)
	hiddenSize := 384

	inputIds, _ := ort.NewTensor(shape, toInt64(en.Ids))
	defer inputIds.Destroy()
	mask, _ := ort.NewTensor(shape, repeatInt64(1, len(en.Ids)))
	defer mask.Destroy()
	types, _ := ort.NewTensor(shape, repeatInt64(0, len(en.Ids)))
	defer types.Destroy()

	outputData := make([]float32, 1*seqLen*int64(hiddenSize))
	outputTensor, _ := ort.NewTensor(ort.NewShape(1, seqLen, int64(hiddenSize)), outputData)
	defer outputTensor.Destroy()

	se.Session.Run([]ort.Value{inputIds, mask, types}, []ort.Value{outputTensor})
	return meanPooling(outputData, hiddenSize)
}

func meanPooling(embeddings []float32, hiddenSize int) []float32 {
	numTokens := len(embeddings) / hiddenSize
	vec := make([]float32, hiddenSize)
	for i := 0; i < numTokens; i++ {
		for h := 0; h < hiddenSize; h++ {
			vec[h] += embeddings[i*hiddenSize+h]
		}
	}
	for h := 0; h < hiddenSize; h++ {
		vec[h] /= float32(numTokens)
	}
	return vec
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, nA, nB float32
	for i := range a {
		dot += a[i] * b[i]
		nA += a[i] * a[i]
		nB += b[i] * b[i]
	}
	return dot / (float32(math.Sqrt(float64(nA))) * float32(math.Sqrt(float64(nB))))
}

func toInt64(in []int) []int64 {
	out := make([]int64, len(in)); for i, v := range in { out[i] = int64(v) }; return out
}

func repeatInt64(v int64, n int) []int64 {
	out := make([]int64, n); for i := range out { out[i] = v }; return out
}
