import unittest

import torch
from sympy.polys.subresultants_qq_zz import res

import cs336_basics.model as model
import cs336_basics.functions as functions
import torch.nn.functional as F

from cs336_basics import optimizers
from cs336_basics.training import train, load_tokenizer, init_tokenizer, tokenize_file_and_persistent
from cs336_basics.optimizers import AdamW


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_model_parameters(self):
        transformer_lm = model.TransformerLM(50257, 1024, 1600, 48, 25, 4288,
                                             10000)
        total_parameter_count = 0
        memory_size = 0

        for parameter in transformer_lm.parameters():
            total_parameter_count += parameter.numel()
            memory_size += parameter.numel() * parameter.element_size()

        print(
            f"parameters: {total_parameter_count:,}, "
            f"memory: {memory_size:,} bytes "
            f"({memory_size / 1024 ** 2:.2f} MiB)"
        )
        print(f"the model FLOPs: {transformer_lm.count_op_flops(torch.randn(1, 1024), 50257, 1600)}, ")

    def test_cross_entropy_loss(self):
        inputs = torch.tensor(
            [
                [
                    [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                    [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                    [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                    [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
                ],
                [
                    [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                    [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                    [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                    [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
                ],
            ]
        )
        labels = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
        # result = functions.cross_entropy_loss(inputs, labels)
        result = functions.cross_entropy_loss(inputs.view(-1, inputs.size(-1)), labels.view(-1))
        expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), labels.view(-1))
        print(expected)
        print(result)

    def test_toy_sgd(self):
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = optimizers.SGD([weights], lr=1)
        for t in range(100):
            opt.zero_grad()
            loss = (weights ** 2).mean()  # compute  a scalar loss value
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients
            opt.step()

    def test_training(self):
        training_data_file = "tests/fixtures/tinystories_sample.txt"
        special_tokens = ["<|endoftext|>"]

        vocab_size = 1000
        context_length = 256

        d_model = 512
        #d_model = 64
        d_ff = 1344
        num_layers = 4
        num_heads = 16
        head_dim = 32

        rope_theta = 10_000

        batch_size = 64
        total_steps = 20_000
        # total_tokens = 327_680_000

        device = "mps"
        transformer_lm = model.TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model,
                                             d_ff=d_ff, num_layers=num_layers, num_heads=num_heads, theta=rope_theta,device=device)
        optimizer = AdamW(transformer_lm.parameters(),device = device)
        tokenizer = init_tokenizer(training_data_file, vocab_size, special_tokens)
        token_file = tokenize_file_and_persistent(training_data_file, special_tokens[0], tokenizer)
        train(dataset_file=token_file, module=transformer_lm, optimizer=optimizer, batch_size=batch_size,
              seq_len=context_length, total_steps=total_steps,device=device)


if __name__ == '__main__':
    unittest.main()
