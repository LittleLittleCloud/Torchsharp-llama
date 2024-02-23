// See https://aka.ms/new-console-template for more information

using FluentAssertions;
using LLAMA;
using TorchSharp;
using System.Runtime.InteropServices;
var vocabPath = @"vocab.json";
var mergesPath = @"merges.txt";
var tokenizer = new BPETokenizer(vocabPath, mergesPath);

// update the following path to where you download the model
var checkpointDirectory = "/home/xiaoyuz/Llama-2-7b";
var device = "cuda";

if (device == "cuda")
{
    // Comment out the following two line if you use a torchsharp runtime package.
    var libTorch = "/anaconda/envs/py38_default/lib/python3.8/site-packages/torch/lib/libtorch.so";
    NativeLibrary.Load(libTorch);
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

torch.manual_seed(100);
var model = LLaMA.Build(
       modelFolder: checkpointDirectory,
       tokenizer: tokenizer,
       maxSeqLen: 128,
       maxBatchSize: 1,
       device: device);

var prompts = new[]
{
    "I believe the meaning of life is",
};
var result = model.TextCompletion(prompts, temperature: 0, echo: true, device: device);

foreach (var item in result)
{
    Console.WriteLine($"generation: {item.generation}");
}
