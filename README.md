## Torchsharp LLaMA

Inspired by [pytorch-llama](https://github.com/hkproj/pytorch-llama), this project implements LLaMA 2 from scratch with [TorchSharp](https://github.com/dotnet/TorchSharp)

## Prerequisites
- git lfs
- .NET 6.0 SDK
- Access to one of LLaMA 2 models

## How to run

- Download the model weight. The model weigh is available from huggingface model hub.
  - llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b
  - llama-2-7b-chat: https://huggingface.co/meta-llama/Llama-2-7b-chat

> [!NOTE]
> Please download the pth version (the one without -hf prefix)

- Change the path in [`Program.cs`](./Program.cs#L12) to the folder where you download the model weight.
- Determine the right torchsharp runtime nuget package on your platform.
  - use `TorchSharp-cuda-linux` if you are on linux and have a nvidia gpu
  - use `TorchSharp-cuda-windows` if you are on windows and have a nvidia gpu
  - use `TorchSharp-cpu` if you don't have a nvidia gpu
- Run the project using `dotnet run`

## About tokenizer
This project uses a BPE tokenizer from [`Microsoft.ML.Tokenizer`](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers.tokenizer?view=ml-dotnet-preview) to tokenize the input text. You can find the `vocab.json` and `merges.txt` under [torcharp-llama](Torchsharp-llama). To use a third-party tokenizer, you can simply replace the `vocab.json` and `merges.txt` with your own tokenizer files.

## Disclaimer
This project is only tested with LLaMA-2-7B model. I do hope I can have the chance to test it with other models, but unfortunately 7B model is already the largest model I can afford to run on my machine. If you have chance to test other models, please let me know if it works or not. Thanks!

Also, this project doesn't come with any warranty. Use it at your own risk.

## TODO
- [ ] Add support to load from `.safetensor` and native ckpt file so that we don't need to convert the model to torchsharp format. The support for `.safetensor` should be an easy one, but the support for native ckpt file is a bit tricky (otherwise why torchsharp format exists in the first place)
- [ ] Add support for lora training
