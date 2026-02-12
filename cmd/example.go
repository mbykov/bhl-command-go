package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/chzyer/readline"
	"github.com/mbykov/bhl-command-go"
)

func main() {
	engine, err := command.NewSearchEngine(
		"../Models/multilingual-e5-small/onnx/model.onnx",
		"../Models/multilingual-e5-small/tokenizer.json",
		"/home/michael/go/ort/lib/libonnxruntime.so",
		0.92, // рекомендованный порог
	)
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	if err := engine.LoadCommands("./data/commands-syn.json"); err != nil {
		log.Fatal(err)
	}

	fmt.Println("✅ Модель загружена, команды проиндексированы.")
	fmt.Println("📜 История команд сохраняется (стрелки вверх/вниз).")
	fmt.Println("Введите фразу (минимум 2 слова) или 'exit' для выхода:")

	rl, err := readline.New("> ")
	if err != nil {
		log.Fatal(err)
	}
	defer rl.Close()

	for {
		line, err := rl.Readline()
		if err != nil { // EOF, Ctrl+D, Ctrl+C
			break
		}
		phrase := strings.TrimSpace(line)
		if phrase == "" {
			continue
		}
		if phrase == "exit" || phrase == "quit" {
			fmt.Println("Выход.")
			break
		}

		result, err := engine.FindCommand(phrase)
		if err != nil {
			fmt.Printf("❌ Ошибка: %v\n", err)
			continue
		}
		if result == nil {
			// if len(strings.Fields(phrase)) < 2 {
			// 	fmt.Printf("❌ Слишком короткая фраза (менее 2 слов).\n")
			// } else {
			// 	fmt.Printf("❌ Команда не найдена.\n")
			// }
            fmt.Printf("❌ Команда не найдена.\n")
		} else {
			fmt.Printf("✅ Найдена команда: %s\n", result.Name)
			fmt.Printf("   Синонимы: %v\n", result.Synonyms)
			fmt.Printf("   Внешняя: %v\n", result.External)
			fmt.Printf("   Сходство: %.4f\n", result.Score)
		}
	}
}
