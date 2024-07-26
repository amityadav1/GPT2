import tiktoken
import torch
import numpy as np
import os

#Simple Data Loader Class
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, datadir, master_process):
    # Load data from the tinyshakesphere data downloaded into input.txt file
        # with open("input.txt", 'r') as file:
        #     text = file.read()
        # enc = tiktoken.get_encoding("gpt2")
        # self.tokens = torch.tensor(enc.encode(text))
    
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
    
        # Load all the shards of data
        assert split in {'train', 'val'}
        data_root = datadir
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for {split}"
        if master_process:
            print(f"Loaded {len(shards)} shards")

        self.current_shard_index = 0
        self.current_position = self.B * self.T * self.process_rank
        self.tokens = self.load_token(self.shards[self.current_shard_index])
        
        # print(f"DataLoader Init: Num tokens: {len(self.tokens)}")
        # print(f"DataLoader Init: Num Batches: {len(self.tokens) // (self.B  * self.T)}")

    def load_token(filename):
        npt = np.load(filename)
        ptt = torch.tensor(np, dtype=torch.long)
        return ptt
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += (B * T * self.num_processes)
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.current_position = B * T * self.process_rank
        return x, y