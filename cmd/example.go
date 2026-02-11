package main

import (
    "fmt"
    "log"

    "github.com/mbykov/bhl-command-go"
)

func main() {
    engine, err := command.NewSearchEngine(
        "../Models/multilingual-e5-small/onnx/model.onnx",
        "../Models/multilingual-e5-small/tokenizer.json",
        "/home/michael/go/ort/lib/libonnxruntime.so",
        0.80,
    )
    if err != nil {
        log.Fatal(err)
    }
    defer engine.Close()

    if err := engine.LoadCommands("./data/commands-syn.json"); err != nil {
        log.Fatal(err)
    }

    phrases := []string{
        "стоп запись",
        "стоп машина",
        "прекратить запись",
        "стоп текст",
        "начать записывать",
        "начать работу",
        "убей запись",
        "сколько времени",
        "неизвестная фраза",
        "покажи латех",
        "съешь латех",
        "формула латех",
        "сформируй латех",
        "нарисуй латех",
    }

    for _, phrase := range phrases {
        result, err := engine.FindCommand(phrase)
        if err != nil {
            fmt.Printf("Error for %q: %v\n", phrase, err)
            continue
        }
        if result == nil {
            fmt.Printf("No match for: %q\n", phrase)
        } else {
            fmt.Printf("example: %q\n  -> command: %s, текст: %s,  Score: %.4f\n", //, External: %v
                phrase, result.Name, result.Synonyms, result.Score) // , result.External
        }
    }
}
