﻿using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;

namespace LLAMA;

public static class Utils
{
    public static void Peek(this Tensor tensor, string id, int n = 10)
    {
        var device = tensor.device;
        tensor = tensor.cpu();
        var shapeString = string.Join(',', tensor.shape);
        var dataString = string.Join(',', tensor.reshape(-1)[..n].to_type(ScalarType.Float32).data<float>().ToArray());
        var tensor_1d = tensor.reshape(-1);
        var tensor_index = torch.arange(tensor_1d.shape[0], dtype: ScalarType.Float32).to(tensor_1d.device).sqrt();
        var avg = (tensor_1d * tensor_index).sum();
        avg = avg / tensor_1d.sum();
        Console.WriteLine($"{id}: sum: {avg.ToSingle()}  dtype: {tensor.dtype} shape: [{shapeString}] device: {device} has grad? {tensor.requires_grad}");
    }

    public static void Peek(this nn.Module model)
    {
        var state_dict = model.state_dict();
        // preview state_dict
        foreach (var (key, value) in state_dict.OrderBy(x => x.Key))
        {
            value.Peek(key);
        }
    }

    public static Tensor ApplyRotaryEmbeddings(Tensor input, Tensor freqsComplex)
    {
        // Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
        // Two consecutive values will become a single complex number
        // (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
        var input_complex = input.to_type(ScalarType.Float32).reshape(input.shape[0], input.shape[1], input.shape[2], -1, 2).view_as_complex();

        // Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        // (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        var freqs_complex_reshaped = freqsComplex.unsqueeze(0).unsqueeze(2);

        // Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        // Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        // (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        var rotated_complex = input_complex * freqs_complex_reshaped;
        // Console.WriteLine(rotated_complex.mean().ToSingle());

        // Convert the complex number back to the real number
        // (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        var rotated = rotated_complex.view_as_real();

        // (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        var rotated_reshaped = rotated.reshape(rotated.shape[0], rotated.shape[1], rotated.shape[2], -1);

        input.shape.Should().BeEquivalentTo(rotated_reshaped.shape);
        return rotated_reshaped.type_as(input);
    }

//    def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
//    # As written in the paragraph 3.2.2 of the paper
//    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
//    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
//    # Build the theta parameter
//    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
//    # Shape: (Head_Dim / 2)
//    theta_numerator = torch.arange(0, head_dim, 2).float ()
//# Shape: (Head_Dim / 2)
//    theta = 1.0 / (theta * *(theta_numerator / head_dim)).to(device) # (Dim / 2)
//    # Construct the positions (the "m" parameter)
//    # Shape: (Seq_Len)
//    m = torch.arange(seq_len, device=device)
//    # Multiply each theta by each position using the outer product.
//    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
//    freqs = torch.outer(m, theta).float ()
//# We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
//# (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
//    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
//    return freqs_complex

    public static Tensor PrecomputeThetaPosFrequencies(int headDim, int seqLen, float theta = 10000.0f)
    {
        // As written in the paragraph 3.2.2 of the paper
        // >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        if (headDim % 2 != 0)
        {
            throw new ArgumentException("Dimension must be divisible by 2", nameof(headDim));
        }

        // Build the theta parameter
        // According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        // Shape: (Head_Dim / 2)
        var thetaNumerator = torch.arange(0, headDim, 2).to(torch.float32);
        // Shape: (Head_Dim / 2)
        var thetaInput = torch.pow(theta, -1.0f * (thetaNumerator / headDim)); // (Dim / 2)
        // Construct the positions (the "m" parameter)
        // Shape: (Seq_Len)
        var m = torch.arange(seqLen);
        // Multiply each theta by each position using the outer product.
        // Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqs = torch.outer(m, thetaInput).to(torch.float32);

        // We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        // (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqsComplex = torch.polar(torch.ones_like(freqs), freqs);

        return freqsComplex;
    }


    public static Tensor RepeatKV(Tensor x, int nRep)
    {
        var batchSize = x.shape[0];
        var seqLen = x.shape[1];
        var nKVHeads = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1)
        {
            return x;
        }

        return x.unsqueeze(3)
                .expand(batchSize, seqLen, nKVHeads, nRep, headDim)
                .reshape(batchSize, seqLen, nKVHeads * nRep, headDim);
    }

}
