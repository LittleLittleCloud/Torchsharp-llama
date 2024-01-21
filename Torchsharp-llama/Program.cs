// See https://aka.ms/new-console-template for more information

using LLAMA;
var vocabPath = @"vocab.json";
var mergesPath = @"merges.txt";
var tokenizer = new BPETokenizer(vocabPath, mergesPath);
var checkpointDirectory = "C:\\Users\\xiaoyuz\\source\\repos\\llama\\llama-2-7b";

var model = LLaMA.Build(
       checkPointsDirectory: checkpointDirectory,
       tokenizer: tokenizer,
       maxSeqLen: 1024,
       maxBatchSize: 1,
       device: "cpu");



var prompts = new[]
{
    "What's the capital of France?",
};
var result = model.TextCompletion(prompts);

foreach (var item in result)
{
    Console.WriteLine($"generation: {item.generation}");
}
