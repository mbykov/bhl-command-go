package command

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

type CommandMapping struct {
	Name     string   `json:"name"`
	Synonyms []string `json:"synonyms"`
	External bool     `json:"external"`
	Score    float32  `json:"score,omitempty"`
}

type synonymEntry struct {
	commandIdx int
	embedding  []float32
}

type SearchEngine struct {
	Tokenizer *tokenizer.Tokenizer
	Session   *ort.DynamicAdvancedSession

	commands   []CommandMapping
	synonyms   []synonymEntry
	threshold  float32
	hiddenSize int
}

// NewSearchEngine — onnxPath: model.onnx, tokenizerPath: tokenizer.json, libPath: libonnxruntime.so
func NewSearchEngine(onnxPath, tokenizerPath, libPath string, threshold float32) (*SearchEngine, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init ORT: %w", err)
	}

	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(onnxPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	se := &SearchEngine{
		Tokenizer: tk,
		Session:   session,
		threshold: threshold,
	}

	if err := se.determineHiddenSize(); err != nil {
		se.Close()
		return nil, fmt.Errorf("determine hidden size: %w", err)
	}

	return se, nil
}

// determineHiddenSize — вызывает модель с коротким текстом, чтобы узнать размер эмбеддинга.
func (se *SearchEngine) determineHiddenSize() error {
	enc, err := se.Tokenizer.EncodeSingle("test")
	if err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	if len(enc.Ids) == 0 {
		return fmt.Errorf("empty token sequence")
	}

	seqLen := int64(len(enc.Ids))
	shape := ort.NewShape(1, seqLen)

	inputIds, _ := ort.NewTensor(shape, toInt64(enc.Ids))
	defer inputIds.Destroy()
	mask, _ := ort.NewTensor(shape, repeatInt64(1, int(seqLen)))
	defer mask.Destroy()
	types, _ := ort.NewTensor(shape, repeatInt64(0, int(seqLen)))
	defer types.Destroy()

	outputData := make([]float32, 1*seqLen*384)
	outputTensor, _ := ort.NewTensor(ort.NewShape(1, seqLen, 384), outputData)
	defer outputTensor.Destroy()

	if err := se.Session.Run([]ort.Value{inputIds, mask, types}, []ort.Value{outputTensor}); err != nil {
		return fmt.Errorf("inference: %w", err)
	}

	shapeActual := outputTensor.GetShape()
	if len(shapeActual) != 3 {
		return fmt.Errorf("unexpected shape: %v", shapeActual)
	}
	se.hiddenSize = int(shapeActual[2])
	return nil
}

// Close — освобождает ресурсы ORT.
func (se *SearchEngine) Close() error {
	var err error
	if se.Session != nil {
		err = se.Session.Destroy()
	}
	ort.DestroyEnvironment()
	return err
}

// LoadCommands — читает JSON, добавляет префикс "passage: " к каждому синониму,
// вычисляет и нормализует эмбеддинги.
func (se *SearchEngine) LoadCommands(jsonPath string) error {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	var cmds []CommandMapping
	if err := json.Unmarshal(data, &cmds); err != nil {
		return fmt.Errorf("unmarshal JSON: %w", err)
	}

	se.commands = cmds
	se.synonyms = nil

	for cmdIdx, cmd := range cmds {
		for _, syn := range cmd.Synonyms {
			if strings.TrimSpace(syn) == "" {
				continue
			}
			prefixed := "passage: " + syn
			emb, err := se.getEmbedding(prefixed)
			if err != nil {
				return fmt.Errorf("embedding for %q: %w", syn, err)
			}
			se.synonyms = append(se.synonyms, synonymEntry{
				commandIdx: cmdIdx,
				embedding:  normalize(emb),
			})
		}
	}
	return nil
}

// FindCommand — поиск команды по фразе.
// Возвращает nil, если фраза состоит менее чем из двух слов.
func (se *SearchEngine) FindCommand(phrase string) (*CommandMapping, error) {
	// 1. Проверка на минимальное количество слов (2+)
	if len(strings.Fields(phrase)) < 2 {
		return nil, nil
	}

	if len(se.synonyms) == 0 {
		return nil, fmt.Errorf("no commands loaded")
	}

	prefixedPhrase := "query: " + phrase
	queryEmb, err := se.getEmbedding(prefixedPhrase)
	if err != nil {
		return nil, fmt.Errorf("query embedding: %w", err)
	}
	queryNorm := normalize(queryEmb)

	var bestScore float32 = -1.0
	var bestIdx = -1

	for i, entry := range se.synonyms {
		score := dotProduct(queryNorm, entry.embedding)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx == -1 || bestScore < se.threshold {
		return nil, nil
	}

	cmd := se.commands[se.synonyms[bestIdx].commandIdx]
	return &CommandMapping{
		Name:     cmd.Name,
		Synonyms: cmd.Synonyms,
		External: cmd.External,
		Score:    bestScore,
	}, nil
}

// --- Внутренние утилиты ------------------------------------------------------

func (se *SearchEngine) getEmbedding(text string) ([]float32, error) {
	enc, err := se.Tokenizer.EncodeSingle(text)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	if len(enc.Ids) == 0 {
		return nil, fmt.Errorf("empty tokens")
	}

	seqLen := int64(len(enc.Ids))
	shape := ort.NewShape(1, seqLen)

	inputIds, _ := ort.NewTensor(shape, toInt64(enc.Ids))
	defer inputIds.Destroy()
	mask, _ := ort.NewTensor(shape, repeatInt64(1, int(seqLen)))
	defer mask.Destroy()
	types, _ := ort.NewTensor(shape, repeatInt64(0, int(seqLen)))
	defer types.Destroy()

	outputData := make([]float32, 1*seqLen*int64(se.hiddenSize))
	outputTensor, _ := ort.NewTensor(ort.NewShape(1, seqLen, int64(se.hiddenSize)), outputData)
	defer outputTensor.Destroy()

	if err := se.Session.Run([]ort.Value{inputIds, mask, types}, []ort.Value{outputTensor}); err != nil {
		return nil, fmt.Errorf("run: %w", err)
	}

	return meanPooling(outputData, se.hiddenSize), nil
}

func meanPooling(emb []float32, hiddenSize int) []float32 {
	numTokens := len(emb) / hiddenSize
	vec := make([]float32, hiddenSize)
	for i := 0; i < numTokens; i++ {
		for h := 0; h < hiddenSize; h++ {
			vec[h] += emb[i*hiddenSize+h]
		}
	}
	for h := 0; h < hiddenSize; h++ {
		vec[h] /= float32(numTokens)
	}
	return vec
}

func normalize(vec []float32) []float32 {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm == 0 {
		return vec
	}
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = v / norm
	}
	return out
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func toInt64(in []int) []int64 {
	out := make([]int64, len(in))
	for i, v := range in {
		out[i] = int64(v)
	}
	return out
}

func repeatInt64(v int64, n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = v
	}
	return out
}
