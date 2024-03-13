// See https://aka.ms/new-console-template for more information

using FluentAssertions;
using LLAMA;
using TorchSharp;
var vocabPath = @"vocab.json";
var mergesPath = @"merges.txt";
var tokenizer = new BPETokenizer(vocabPath, mergesPath);

var prompts = new[]
{
    "I believe the meaning of life is",
    "Replace this text in the input field to see how llama tokenization works."
};

foreach (var prompt in prompts)
{
    var tokens = tokenizer.Encode(prompt, bos: true, eos: false);
    var decoded = tokenizer.Decode(tokens);
    Console.WriteLine($"Prompt: {prompt}");
    Console.WriteLine($"Tokens: {string.Join(", ", tokens)}");
    Console.WriteLine($"Decoded: {decoded}");
    Console.WriteLine();
}
