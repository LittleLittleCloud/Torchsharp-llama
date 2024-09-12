using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace LLAMA;

public interface ITokenizer
{
    public int[] Encode(string input, bool bos, bool eos);

    public string Decode(int[] input);

    public int VocabSize { get; }

    public int PadId { get; }

    public int BosId { get; }

    public int EosId { get; }
}

public class Norm : Normalizer
{
    public override NormalizedString Normalize(string original)
    {
        // replace space with _
        var normalized = original.Replace(" ", "▁");

        return new NormalizedString(original, normalized, null, isOneToOneMapping: true);
    }
}

public class TikTokenNormalizer : Normalizer
{
    public override NormalizedString Normalize(string original)
    {
        // replace newline with Ċ
        var normalized = original.Replace(Environment.NewLine, "Ċ");
        // replace whitespace with Ġ
        normalized = normalized.Replace(' ', 'Ġ');

        return new NormalizedString(original, normalized, null, isOneToOneMapping: true);
    }
}

public class PreTokenizer : Microsoft.ML.Tokenizers.PreTokenizer
{
    public override IReadOnlyList<Split> PreTokenize(string sentence)
    {
        var split = new Split(sentence, new(0, sentence.Length));

        return new List<Split> { split };
    }
}

public class SplitPreTokenizer : Microsoft.ML.Tokenizers.PreTokenizer
{
    private readonly string _pattern;

    public SplitPreTokenizer(string pattern)
    {
        this._pattern = pattern;
    }

    public override IReadOnlyList<Split> PreTokenize(string? sentence)
    {
        if (sentence == null)
        {
            return [];
        }

        List<Split> list = new List<Split>();
        foreach (Match item in Regex.Matches(sentence, _pattern))
        {
            list.Add(new Split(item.Value, (item.Index, item.Index + item.Length)));
        }

        return list;
    }
}

public class TokenizeDecoder : Microsoft.ML.Tokenizers.TokenizerDecoder
{
    private char spaceReplacement = '▁';
    private char newlineReplacement = 'Ċ';
    private string bos = "<s>";
    private string eos = "</s>";

    public TokenizeDecoder(string bos = "<s>", string eos = "</s>", char spaceReplacement = '▁', char newlineReplacement = 'Ċ')
    {
        this.bos = bos;
        this.eos = eos;
        this.spaceReplacement = spaceReplacement;
        this.newlineReplacement = newlineReplacement;
    }

    public override string Decode(IEnumerable<string> tokens)
    {
        var str = string.Join("", tokens);
        str = str.Replace(spaceReplacement, ' ');
        str = str.Replace(newlineReplacement.ToString(), Environment.NewLine);

        if (str.StartsWith(bos))
        {
            str = str.Substring(bos.Length);
        }

        if (str.EndsWith(eos))
        {
            str = str.Substring(0, str.Length - eos.Length);
        }

        return str;
    }
}

public class LLama2Tokenizer : ITokenizer
{
    private Tokenizer tokenizer;
    private bool addPrecedingSpace;

    public LLama2Tokenizer(string vocabPath, string mergesPath, bool addPrecedingSpace = true, int padToken = -1, int startToken = 1, int endToken = 2)
    {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;
        var bpe = new Bpe(vocabPath, mergesPath);
        this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new Norm());
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
        this.tokenizer.Decoder = decoder;
    }

    public LLama2Tokenizer(Dictionary<string, int> vocab, List<string> merges, bool addPrecedingSpace = true, int padToken = -1, int startToken = 1, int endToken = 2)
    {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;
        // save vocab to vocab-temp.json
        var vocabTempPath = "vocab-temp.json";
        var json = JsonSerializer.Serialize(vocab);
        File.WriteAllText(vocabTempPath, json);

        // save merges to merges-temp.txt
        var mergesTempPath = "merges-temp.txt";
        File.WriteAllLines(mergesTempPath, merges);

        var bpe = new Bpe(vocabTempPath, mergesTempPath);
        
        this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new Norm());
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
        this.tokenizer.Decoder = decoder;

        // delete temp files
        File.Delete(vocabTempPath);
        File.Delete(mergesTempPath);
    }

    public static LLama2Tokenizer FromPretrained(
        string folder,
        string tokenizerJsonPath = "tokenizer.json"
    )
    {
        tokenizerJsonPath = Path.Combine(folder, tokenizerJsonPath);
        var json = File.ReadAllText(tokenizerJsonPath);
        var jsonDocument = JsonDocument.Parse(json);
        // vocab: .model.vocab
        var vocabNode = jsonDocument.RootElement.GetProperty("model").GetProperty("vocab");

        // to Dictionary<string, int>
        var vocab = new Dictionary<string, int>();
        foreach (var item in vocabNode.EnumerateObject())
        {
            vocab[item.Name] = item.Value.GetInt32();
        }

        // added tokens: .added_tokens
        var addedTokensNode = jsonDocument.RootElement.GetProperty("added_tokens");
        foreach (var item in addedTokensNode.EnumerateArray())
        {
            // get id from item.id
            var id = item.GetProperty("id").GetInt32();
            var content = item.GetProperty("content").GetString()!;
            vocab[content] = id;
        }

        // merges: .model.merges
        var mergesNode = jsonDocument.RootElement.GetProperty("model").GetProperty("merges");
        // merges: List<string>
        var merges = new List<string>();
        foreach (var item in mergesNode.EnumerateArray())
        {
            merges.Add(item.GetString()!);
        }

        var startToken = vocab["<|begin_of_text|>"];
        var endToken = vocab["<|end_of_text|>"];

        return new LLama2Tokenizer(vocab, merges, startToken: startToken, endToken: endToken);
    }

    public int VocabSize => this.tokenizer.Model.GetVocabSize();

    public int PadId { get; }

    public int BosId { get; }

    public int EosId { get; }

    public string Decode(int[] input)
    {
        var str = this.tokenizer.Decode(input) ?? throw new Exception("Failed to decode");
        if (this.addPrecedingSpace)
        {
            str = str.TrimStart();
        }

        return str;
    }

    public int[] Encode(string input, bool bos, bool eos)
    {
        if (this.addPrecedingSpace)
        {
            input = " " + input;
        }
        var tokens = this.tokenizer.Encode(input).Ids.ToArray();
        if (bos)
        {
            tokens = new int[] { this.BosId }.Concat(tokens).ToArray();
        }
        if (eos)
        {
            tokens = tokens.Concat(new int[] { this.EosId }).ToArray();
        }

        Console.WriteLine($"tokens: {string.Join(",", tokens)}");

        return tokens;
    }
}

public class LLama3Tokenizer : ITokenizer
{
    private Tokenizer tokenizer;
    private bool addPrecedingSpace;

    public LLama3Tokenizer(string vocabPath, string mergesPath, bool addPrecedingSpace = false, int padToken = -1, int startToken = 1, int endToken = 2)
    {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;
        var bpe = new Bpe(vocabPath, mergesPath);
        this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new TikTokenNormalizer());
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!, 'Ġ');
        this.tokenizer.Decoder = decoder;
    }

    public LLama3Tokenizer(Dictionary<string, int> vocab, List<string> merges, bool addPrecedingSpace = false, int padToken = -1, int startToken = 1, int endToken = 2)
    {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;
        // save vocab to vocab-temp.json
        var vocabTempPath = "vocab-temp.json";
        var json = JsonSerializer.Serialize(vocab);
        File.WriteAllText(vocabTempPath, json);

        // save merges to merges-temp.txt
        var mergesTempPath = "merges-temp.txt";
        File.WriteAllLines(mergesTempPath, merges);

        var bpe = new Bpe(vocabTempPath, mergesTempPath);
        this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new TikTokenNormalizer());
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!, 'Ġ');
        this.tokenizer.Decoder = decoder;

        // delete temp files
        File.Delete(vocabTempPath);
        File.Delete(mergesTempPath);
    }

    public static LLama3Tokenizer FromPretrained(
        string folder,
        string tokenizerJsonPath = "tokenizer.json"
    )
    {
        tokenizerJsonPath = Path.Combine(folder, tokenizerJsonPath);
        var json = File.ReadAllText(tokenizerJsonPath);
        var jsonDocument = JsonDocument.Parse(json);
        // vocab: .model.vocab
        var vocabNode = jsonDocument.RootElement.GetProperty("model").GetProperty("vocab");

        // to Dictionary<string, int>
        var vocab = new Dictionary<string, int>();
        foreach (var item in vocabNode.EnumerateObject())
        {
            vocab[item.Name] = item.Value.GetInt32();
        }

        // added tokens: .added_tokens
        var addedTokensNode = jsonDocument.RootElement.GetProperty("added_tokens");
        foreach (var item in addedTokensNode.EnumerateArray())
        {
            // get id from item.id
            var id = item.GetProperty("id").GetInt32();
            var content = item.GetProperty("content").GetString()!;
            vocab[content] = id;
        }

        // merges: .model.merges
        var mergesNode = jsonDocument.RootElement.GetProperty("model").GetProperty("merges");
        // merges: List<string>
        var merges = new List<string>();
        foreach (var item in mergesNode.EnumerateArray())
        {
            merges.Add(item.GetString()!);
        }

        var startToken = vocab["<|begin_of_text|>"];
        var endToken = vocab["<|end_of_text|>"];

        return new LLama3Tokenizer(vocab, merges, startToken: startToken, endToken: endToken);
    }

    public int VocabSize => this.tokenizer.Model.GetVocabSize();

    public int PadId { get; }

    public int BosId { get; }

    public int EosId { get; }

    public string Decode(int[] input)
    {
        var str = this.tokenizer.Decode(input) ?? throw new Exception("Failed to decode");
        if (this.addPrecedingSpace)
        {
            str = str.TrimStart();
        }

        return str;
    }

    public int[] Encode(string input, bool bos, bool eos)
    {
        if (this.addPrecedingSpace)
        {
            input = " " + input;
        }
        var tokens = this.tokenizer.Encode(input).Ids.ToArray();
        if (bos)
        {
            tokens = new int[] { this.BosId }.Concat(tokens).ToArray();
        }
        if (eos)
        {
            tokens = tokens.Concat(new int[] { this.EosId }).ToArray();
        }

        Console.WriteLine($"tokens: {string.Join(",", tokens)}");

        return tokens;
    }
}

