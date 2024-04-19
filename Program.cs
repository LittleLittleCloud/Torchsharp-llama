// See https://aka.ms/new-console-template for more information

using FluentAssertions;
using LLAMA;
using TorchSharp;

var tokenizerFolder = @"C:\Users\xiaoyuz\source\repos\Meta-Llama-3-8B\";
var tokenizer = LLama3Tokenizer.FromPretrained(tokenizerFolder);

// update the following path to where you download the model
var checkpointDirectory = @"C:\Users\xiaoyuz\source\repos\llama3\Meta-Llama-3-8B\";
var device = "cuda";

if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

torch.manual_seed(1);
var model = LLaMA.Build(
       modelFolder: checkpointDirectory,
       tokenizer: tokenizer,
       maxSeqLen: 1024,
       maxBatchSize: 1,
       device: device);

var prompts = new[]
{
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that",
    @"A brief message congratulating the team on the launch:

        Hi everyone,

        I just ",
    @"Translate English to French:

    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>",
};

// tokenizer

foreach (var prompt in prompts)
{
    Console.WriteLine($"prompt: {prompt}");
    tokenizer.Encode(prompt, false, false).Should().NotBeNull();
    var result = model.TextCompletion([prompt], echo: true, device: device);

    foreach (var item in result)
    {
        Console.WriteLine($"generation: {item.generation}");
    }
}


