using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.PyBridge;

namespace LLAMA;

public enum Role
{
    System = 0,
    User = 1,
    Assistant = 2,
}
public record CompletionPrediction(string generation, string[]? tokens, float[]? logProbs);

public record Message(Role role, string content);

public record ChatPrediction(Message generation, string[]? tokens, float[]? logProbs);

public class LLaMA
{
    private Transformer transformer;
    private ITokenizer tokenizer;

    public LLaMA(Transformer transformer, ITokenizer tokenizer)
    {
        this.transformer = transformer;
        this.tokenizer = tokenizer;
    }

    public static LLaMA Build(
        string modelFolder,
        ITokenizer tokenizer,
        int maxSeqLen,
        int maxBatchSize,
        string paramJsonPath = "params.json",
        string modelWeightPath = "consolidated.00.pth",
        string device = "cpu")
    {
        var stopWatch = new Stopwatch();
        stopWatch.Start();
        paramJsonPath = Path.Combine(modelFolder, paramJsonPath);
        var modelArgs = JsonSerializer.Deserialize<ModelArgs>(File.ReadAllText(paramJsonPath)) ?? throw new Exception("Failed to deserialize model args");
        modelArgs.VocabSize = 128256;
        modelArgs.MaxSeqLen = maxSeqLen;
        modelArgs.MaxBatchSize = maxBatchSize;
        // print model args
        var modelArgsJson = JsonSerializer.Serialize(modelArgs, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"modelArgs: {modelArgsJson}");
        var checkpointPath = Path.Combine(modelFolder, modelWeightPath);
        var model = new Transformer(modelArgs);
        var loadedParameters = new Dictionary<string, bool>();
        model.load_py(location: checkpointPath, strict: false, loadedParameters: loadedParameters);
        // print loaded parameters
        foreach (var (key, value) in loadedParameters.OrderBy(x => x.Key))
        {
            Console.WriteLine($"loadedParameters: {key} {value}");
        }
        model = model.to(device);
        stopWatch.Stop();
        Console.WriteLine($"Loading checkpoint took {stopWatch.ElapsedMilliseconds} ms");

        return new LLaMA(model, tokenizer);
    }

    public (int[][], float[][]?) Generate(
        int[][] promptTokens,
        int maxGenLen,
        float temperature = 0.6f,
        float topP = 0.9f,
        bool logProbs = false,
        bool echo = false,
        string device = "cpu")
    {
        torch.Tensor? tokenLogProbs = null;
        var batch = promptTokens.Length;
        var param = this.transformer.Args;
        batch.Should().BeLessThanOrEqualTo(param.MaxBatchSize, "Batch size should be less than or equal to the max batch size");

        var minPromptLen = promptTokens.Min(x => x.Length);
        var maxPromptLen = promptTokens.Max(x => x.Length);
        maxPromptLen.Should().BeLessThanOrEqualTo(param.MaxSeqLen, "Prompt length should be less than or equal to the max sequence length");

        var totalLen = Math.Min(maxPromptLen + maxGenLen, param.MaxSeqLen);

        var tokens = torch.full(new long[] {batch, totalLen}, this.tokenizer.PadId, dtype: torch.int64, device: device);
        for (var i = 0; i < batch; i++)
        {
            var promptLen = promptTokens[i].Length;
            tokens[i, 0..promptLen] = torch.tensor(promptTokens[i], dtype: torch.int64, device: device);
        }

        if (logProbs)
        {
            tokenLogProbs = torch.zeros(batch, totalLen, this.tokenizer.VocabSize, dtype: torch.float32, device: device);
        }

        using (var _ = torch.no_grad())
        {
            var prevPos = 0;
            var eosReached = torch.tensor(new bool[batch], device: device);
            var inputTextMask = tokens != this.tokenizer.PadId;

            torch.Tensor logits;
            if (minPromptLen == totalLen)
            {
                logits = this.transformer.forward(tokens, prevPos);
                tokenLogProbs = -torch.nn.functional.cross_entropy(input: logits.transpose(1, 2), target: tokens, reduction: torch.nn.Reduction.None, ignore_index: this.tokenizer.PadId);
            }

            for (int curPos = minPromptLen; curPos != totalLen; curPos++)
            {
                logits = this.transformer.forward(tokens[.., prevPos..curPos], prevPos);
                torch.Tensor nextToken;
                if (temperature > 0)
                {
                    var probs = torch.softmax(logits[.., -1] / temperature, dim: -1);
                    nextToken = this.SampleTopP(probs, topP);
                }
                else
                {
                    nextToken = torch.argmax(logits[.., -1], dim: -1);
                }

                nextToken = nextToken.reshape(-1);
                // # only replace token if prompt has already been generated
                nextToken = torch.where(inputTextMask[.., curPos], tokens[.., curPos], nextToken);

                // print nextToken
                Console.WriteLine($"nextToken: {string.Join(",", nextToken.data<long>())}");
                tokens[.., curPos] = nextToken;
                if (logProbs)
                {
                    tokenLogProbs![.., (prevPos + 1) .. (curPos + 1)] = - torch.nn.functional.cross_entropy(input: logits.transpose(1, 2), target: tokens[.., (prevPos + 1) .. (curPos + 1)], reduction: torch.nn.Reduction.None, ignore_index: this.tokenizer.PadId);
                }

                eosReached |= (~inputTextMask[.., curPos]) & (nextToken == this.tokenizer.EosId);
                if (eosReached.all().item<bool>())
                {
                    break;
                }

                prevPos = curPos;
            }

            var outputTokens = new int[batch][];
            var outputLogProbs = new float[batch][];

            for (var i = 0; i < batch; i++)
            {
                // cut to max gen len
                var start = echo ? 0 : promptTokens[i].Length;
                var toks = tokens[i][start..(promptTokens[i].Length + maxGenLen)].data<long>().Select(x => (int)x).ToArray();
                float[]? probs = null;
                if (logProbs)
                {
                    probs = tokenLogProbs![i][start..(promptTokens[i].Length + maxGenLen)].data<float>().ToArray();
                }

                // cut to first eos if any
                if (toks.Contains(this.tokenizer.EosId))
                {
                    var eosPos = Array.IndexOf(toks, this.tokenizer.EosId);
                    toks = toks[..eosPos];
                    if (logProbs)
                    {
                        probs = probs![..eosPos];
                    }
                }

                outputTokens[i] = toks;
                if (logProbs)
                {
                    outputLogProbs[i] = probs!;
                }
            }

            return (outputTokens, logProbs ? null : outputLogProbs);
        }
    }

    public CompletionPrediction[] TextCompletion(
        string[] prompts,
        int? maxGenLen = null,
        float temperature = 0.6f,
        float topP = 0.9f,
        bool logProbs = false,
        bool echo = false,
        string device = "cpu")
    {
        if (maxGenLen == null)
        {
            maxGenLen = this.transformer.Args.MaxSeqLen - 1;
        }

        var prompTokens = prompts.Select(x => this.tokenizer.Encode(x, bos: true, eos: false)).ToArray();
        var (outputTokens, outputLogProbs) = this.Generate(prompTokens, maxGenLen.Value, temperature, topP, logProbs, echo, device);
        return outputTokens.Select((x, i) => new CompletionPrediction(this.tokenizer.Decode(x), x.Select(x => this.tokenizer.Decode([x])).ToArray(), logProbs ? outputLogProbs![i] : null)).ToArray();
    }

    private torch.Tensor SampleTopP(torch.Tensor logits, float topP)
    {
        (var probsSort, var probsIndex) = torch.sort(logits, dim: -1, descending: true);
        var cumsum = torch.cumsum(probsSort, dim: -1);
        var mask = cumsum - probsSort > topP;
        probsSort[mask] = 0f;
        probsSort /= probsSort.sum(dim: -1, keepdim: true);
        var nextToken = torch.multinomial(probsSort, num_samples: 1);
        nextToken = torch.gather(probsIndex, dim: -1, index: nextToken);
        return nextToken;
    }
}
