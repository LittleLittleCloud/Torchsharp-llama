using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static System.Formats.Asn1.AsnWriter;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
namespace LLAMA;
//class ModelArgs :
//    dim: int = 4096
//    n_layers: int = 32
//    n_heads: int = 32
//    n_kv_heads: Optional[int] = None
//    vocab_size: int = -1 # Later set in the build method
//    multiple_of: int = 256
//    ffn_dim_multiplier: Optional[float] = None
//    norm_eps: float = 1e-5

//    # Needed for KV cache
//    max_batch_size: int = 32
//    max_seq_len: int = 2048

//    device: str = None
public struct ModelArgs
{
    [JsonPropertyName("dim")]
    public int Dim { get; set; } = 4096;

    [JsonPropertyName("n_layers")]
    public int NLayers { get; set; } = 32;

    [JsonPropertyName("n_heads")]
    public int NHeads { get; set; } = 32;

    [JsonPropertyName("n_kv_heads")]
    public int? NKVHeads { get; set; } = null;

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = -1;

    [JsonPropertyName("multiple_of")]
    public int MultipleOf { get; set; } = 256;

    [JsonPropertyName("ffn_dim_multiplier")]
    public float? FFNDimMultiplier { get; set; } = null;

    [JsonPropertyName("norm_eps")]
    public float NormEps { get; set; } = 1e-5f;

    [JsonPropertyName("max_batch_size")]
    public int MaxBatchSize { get; set; } = 3;

    [JsonPropertyName("max_seq_len")]
    public int MaxSeqLen { get; set; } = 1024;

    [JsonPropertyName("device")]
    public string? Device { get; set; } = null;

    public ModelArgs()
    {
    }
}

public class RMSNorm : torch.nn.Module<Tensor, Tensor>
{
    private int _dim;
    private float _eps;
    private Parameter weight;
    public RMSNorm(int dim, float eps = 1e-8f)
        : base(nameof(RMSNorm))
    {
        this._dim = dim;
        this._eps = eps;

        // the gamma scalar
        this.weight = torch.nn.Parameter(torch.ones(dim));
    }

    private Tensor Norm(Tensor x)
    {
        // (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        // rsqrt = 1 / sqrt
        var sqrt = torch.sqrt(torch.mean(x * x, dimensions: [-1L], keepdim: true) + this._eps);
        return x / sqrt;
    }

    public override Tensor forward(Tensor input)
    {
        // (B, Seq_Len, Dim)
        var normed = this.Norm(input);
        // (B, Seq_Len, Dim) * (Dim) = (B, Seq_Len, Dim)
        return normed * this.weight;
    }
}

public class SelfAttention : torch.nn.Module<Tensor, int, Tensor, Tensor?, Tensor>
{
    private int nKVHeads;
    private int nHeadsQ;
    private int nRep;
    private int headDim;
    private Linear wq;
    private Linear wk;
    private Linear wv;
    private Linear wo;
    private Tensor cache_k;
    private Tensor cache_v;

    public SelfAttention(ModelArgs args)
        : base(nameof(SelfAttention))
    {
        // # Indicates the number of heads for the Keys and Values
        this.nKVHeads = args.NKVHeads ?? args.NHeads;
        // Indicates the number of heads for the Queries
        this.nHeadsQ = args.NHeads;
        // Indicates how many times the Keys and Values should be repeated
        this.nRep = this.nHeadsQ / this.nKVHeads;
        //Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        this.headDim = args.Dim / args.NHeads;

        this.wq = torch.nn.Linear(args.Dim, args.NHeads * this.headDim, hasBias: false);
        this.wk = torch.nn.Linear(args.Dim, this.nKVHeads * this.headDim, hasBias: false);
        this.wv = torch.nn.Linear(args.Dim, this.nKVHeads * this.headDim, hasBias: false);
        this.wo = torch.nn.Linear(args.NHeads * this.headDim, args.Dim, hasBias: false);
        RegisterComponents();

        this.cache_k = torch.zeros(args.MaxBatchSize, args.MaxSeqLen, this.nKVHeads, this.headDim);
        this.cache_v = torch.zeros(args.MaxBatchSize, args.MaxSeqLen, this.nKVHeads, this.headDim);
    }

    public override Tensor forward(Tensor input, int startPos, Tensor freqsComplex, Tensor? mask = null)
    {
        int batchSize = (int)input.shape[0];
        int seqLen = (int)input.shape[1];
        var dim = input.shape[2];

        // (B, Seq_Len, Dim) -> (B, Seq_Len, N_Heads * Head_Dim)
        var xq = this.wq.forward(input);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        var xk = this.wk.forward(input);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        var xv = this.wv.forward(input);

        // (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batchSize, seqLen, this.nHeadsQ, this.headDim);

        // (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.view(batchSize, seqLen, this.nKVHeads, this.headDim);

        // (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view(batchSize, seqLen, this.nKVHeads, this.headDim);

        // (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = Utils.ApplyRotaryEmbeddings(xq, freqsComplex);

        // (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = Utils.ApplyRotaryEmbeddings(xk, freqsComplex);

        // replace entries in cache
        this.cache_k[..batchSize, startPos..(startPos + seqLen)] = xk;
        this.cache_v[..batchSize, startPos..(startPos + seqLen)] = xv;

        var keys = this.cache_k[..batchSize, ..(startPos + seqLen)];
        var values = this.cache_v[..batchSize, ..(startPos + seqLen)];

        // Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        // (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = Utils.RepeatKV(keys, this.nRep);

        // (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = Utils.RepeatKV(values, this.nRep);

        // (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2);

        // (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2);

        // (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2);

        // (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
        var scores = torch.matmul(xq, keys.transpose(2, 3)) / Math.Sqrt(this.headDim);

        if (mask is not null)
        {
            scores = scores + mask;
        }

        var softmax = torch.nn.functional.softmax(scores, dim: -1);

        // (B, H_Q, Seq_Len, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        var output = torch.matmul(softmax, values);

        // (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        output = output.transpose(1, 2).contiguous().view(batchSize, seqLen, -1);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        output = this.wo.forward(output);

        return output;
    }
}

public class FeedForward : torch.nn.Module<Tensor, Tensor>
{
    private Linear w1;
    private Linear w2;
    private Linear w3;

    public FeedForward(ModelArgs args)
        : base(nameof(FeedForward))
    {
        var hiddenDim = args.Dim * 4;
        hiddenDim = 2 * hiddenDim / 3;
        hiddenDim = args.FFNDimMultiplier.HasValue ? (int)args.FFNDimMultiplier.Value * hiddenDim : hiddenDim;

        // Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hiddenDim = args.MultipleOf * ((hiddenDim + args.MultipleOf - 1) / args.MultipleOf);

        this.w1 = torch.nn.Linear(args.Dim, hiddenDim, hasBias: false);
        this.w2 = torch.nn.Linear(hiddenDim, args.Dim, hasBias: false);
        this.w3 = torch.nn.Linear(args.Dim, hiddenDim, hasBias: false);
    }

    public override Tensor forward(Tensor input)
    {
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Hidden_Dim)
        var swish = torch.nn.functional.silu(this.w1.forward(input));
        // (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        var xV = this.w3.forward(input);
        // (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Hidden_Dim)
        var x = swish * xV;
        // (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        x = this.w2.forward(x);

        return x;
    }
}

public class EncoderBlock : torch.nn.Module<Tensor, int, Tensor, Tensor?, Tensor>
{
    private SelfAttention attention;
    private FeedForward feed_forward;
    private RMSNorm attention_norm;
    private RMSNorm ffn_norm;

    public EncoderBlock(ModelArgs args)
        : base(nameof(EncoderBlock))
    {
        this.attention = new SelfAttention(args);
        this.feed_forward = new FeedForward(args);
        this.attention_norm = new RMSNorm(args.Dim, eps: args.NormEps);
        this.ffn_norm = new RMSNorm(args.Dim, eps: args.NormEps);
    }

    public override Tensor forward(Tensor input, int startPos, Tensor freqsComplex, Tensor? mask)
    {
        // (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        var x = this.attention_norm.forward(input);
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = this.attention.forward(x, startPos, freqsComplex, mask);
        // (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = x + input;
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = this.ffn_norm.forward(x);
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = this.feed_forward.forward(x);
        // (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = x + input;

        return x;
    }
}

public class Transformer : nn.Module<Tensor, int, Tensor>
{
    private ModelArgs args;
    private int vocabSize;
    private int nLayers;
    private Embedding tok_embeddings;
    private ModuleList<nn.Module<Tensor, int, Tensor, Tensor?, Tensor>> layers;
    private RMSNorm norm;
    private Linear output;
    private Tensor freqs_compex;

    public Transformer(ModelArgs args)
        : base(nameof(Transformer))
    {
        args.VocabSize.Should().BeGreaterThan(0, "Vocab size must be set");

        this.args = args;
        this.vocabSize = args.VocabSize;
        this.nLayers = args.NLayers;
        this.tok_embeddings = nn.Embedding(this.vocabSize, this.args.Dim);

        this.layers = nn.ModuleList<nn.Module<Tensor, int, Tensor, Tensor?, Tensor>>();
        for (int i = 0; i < this.nLayers; i++)
        {
            this.layers.Add(new EncoderBlock(args));
        }

        this.norm = new RMSNorm(args.Dim, eps: args.NormEps);
        this.output = nn.Linear(args.Dim, this.vocabSize, hasBias: false);

        RegisterComponents();

        this.freqs_compex = Utils.PrecomputeThetaPosFrequencies(args.Dim / args.NHeads, args.MaxSeqLen * 2, args.Device);
    }

    public ModelArgs Args => this.args;

    public override Tensor forward(Tensor tokens, int startPos)
    {
        // (B, Seq_Len) -> (B, Seq_Len, Dim)
        var batch = tokens.shape[0];
        var seqLen = (int)tokens.shape[1];

        // print tokens shape
        Console.WriteLine($"tokens shape: {string.Join(",", tokens.shape)}");
        var h = this.tok_embeddings.forward(tokens);
        var freqsComplex = this.freqs_compex[startPos..(startPos + seqLen)];

        Tensor? mask = null;
        if (seqLen > 1)
        {
            mask = torch.full(new long[] {seqLen, seqLen}, value: float.NegativeInfinity, device: this.args.Device);
            // (B, Seq_Len) -> (B, Seq_Len, Seq_Len)
            mask = torch.triu(mask, diagonal: 1);
            // (B, Seq_Len, Seq_Len) -> (B, Seq_Len, Seq_Len)

            var zeros = torch.zeros(seqLen, startPos);
            mask = torch.hstack([zeros, mask]).type_as(h);
        }
        for (int i = 0; i < this.nLayers; i++)
        {
            h = this.layers[i].forward(h, startPos, freqsComplex, mask);
        }

        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = this.norm.forward(h);
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Vocab_Size)
        var output = this.output.forward(h);

        return output;
    }
}
