// See https://aka.ms/new-console-template for more information

using FluentAssertions;
using LLAMA;
using TorchSharp;
using static TorchSharp.torch;
var vocabPath = @"vocab.json";
var mergesPath = @"merges.txt";
var tokenizer = new BPETokenizer(vocabPath, mergesPath);
var checkpointDirectory = "/home/xiaoyuz/llama/llama-2-7b";
var device = "cuda";

if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

torch.manual_seed(100);
var model = LLaMA.Build(
       checkPointsDirectory: checkpointDirectory,
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

// test rms norm
// var modelArgs = new ModelArgs{
//     VocabSize = 32000,
//     Device = "cuda",
//     MaxSeqLen = 10,
//     MaxBatchSize = 1,
//     Dim = 4096,
//     NormEps = 1e-5f,
// };

// var input = torch.arange(0, modelArgs.MaxSeqLen * modelArgs.Dim * modelArgs.Dim, device: device).view(1, modelArgs.MaxSeqLen, modelArgs.Dim, modelArgs.Dim);
// input = input.to_type(ScalarType.BFloat16);
// input.Peek("input");
// var rms = new RMSNorm(modelArgs);
// var output = rms.forward(input);
// output.Peek("output");

// var feedForward = new FeedForward(modelArgs);
// var feedForwardStateDict = feedForward.state_dict();
// feedForwardStateDict.LoadStateDict("/home/xiaoyuz/llama/feedforward.pt");
// feedForward.load_state_dict(feedForwardStateDict);
// feedForward = feedForward.to(device);
// var output2 = feedForward.forward(output);
// output2.Peek("output2");

// input = torch.arange(0, modelArgs.MaxSeqLen * modelArgs.Dim, device: device).view(1, modelArgs.MaxSeqLen, modelArgs.Dim);
// input = input.to_type(ScalarType.Float32);
// var selfAttention = new SelfAttention(modelArgs);
// var selfAttentionStateDict = selfAttention.state_dict();
// selfAttentionStateDict.LoadStateDict("/home/xiaoyuz/llama/selfattention.pt");
// selfAttention.load_state_dict(selfAttentionStateDict);
// selfAttention = selfAttention.to(device);
// var precompute_theta_pos_frequencies = Utils.PrecomputeThetaPosFrequencies(modelArgs.Dim / modelArgs.NHeads, modelArgs.MaxSeqLen, device);
// var output3 = selfAttention.forward(input, 0, precompute_theta_pos_frequencies, null);
// output3.Peek("output3");
