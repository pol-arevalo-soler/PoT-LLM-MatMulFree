import numpy as np
from Bio import SeqIO
from random import randint
import os
import torch
import sys
import h5py



TOKENS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','Y','Z','X',
          '<','>', # start and end of sequence tokens
          '+', # padding token
          '#', # mask token
          ] 

TOKENS_ESM1 = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','Y','Z','X',
          '<','>', # start and end of sequence tokens
          '+', # padding token
          '#', # mask token
          '.'] # sequence separation token

class ProteinTokenizer(object):
    def __init__(self, max_seq_length=1024, masking_rate=0.15):
        self.max_seq_length = max_seq_length
        self.masking_rate = masking_rate
        self.base2index = dict((base, idx) for idx, base in enumerate(TOKENS))
        self.num_tokens = len(set(self.base2index.values()))
        self.aa_tokens = [self.base2index[token] for token in TOKENS[:24]]

    def truncate(self, sequence):
        if sequence.size(dim=0) <= self.max_seq_length:
            return sequence
        j = randint(0, sequence.size(dim=0) - self.max_seq_length)
        return sequence[j:j+self.max_seq_length]

    def mask(self, sequence):
        ground_truth = torch.empty_like(sequence)
        ground_truth.fill_(self.base2index['#'])
        mask = torch.rand(sequence.size()) < self.masking_rate
        mask = mask & (sequence != self.base2index['+']) & (sequence != self.base2index['<']) & (sequence != self.base2index['>'])
        mask_idx = mask.nonzero()
        for i in mask_idx:
            ground_truth[i] = sequence[i]
            p = np.random.rand()
            if p < 0.1:
                sequence[i] = self.aa_tokens[np.random.randint(0, len(self.aa_tokens))]
            elif p < 0.8:
                sequence[i] = self.base2index['#']
        return sequence, ground_truth

    def encode_sequence(self, sequence, sos_eos_tokens=True):
        if sos_eos_tokens:
            encoded_sequence = [self.base2index['<']] + [self.base2index[aa] for aa in sequence] + [self.base2index['>']]
        else:
            encoded_sequence = [self.base2index[aa] for aa in sequence]
        return torch.tensor(encoded_sequence, dtype=torch.long)
    
    def clean_seq_encode(self, sequence, sos_eos_tokens=True):
        clean_sequence = sequence.replace("-","")
        clean_sequence2 = clean_sequence.replace("J","I")
        if sos_eos_tokens:
            clean_sequence = sequence.replace("-","")
            clean_sequence2 = clean_sequence.replace("J","I")
            encoded_sequence = [self.base2index['<']] + [self.base2index[aa] for aa in clean_sequence2] + [self.base2index['>']]
        else:
            encoded_sequence = [self.base2index[aa] for aa in clean_sequence2]
        return torch.tensor(encoded_sequence, dtype=torch.long)
    
    def encode(self, sequence):
        try:
            sequence = self.encode_sequence(sequence)
        except KeyError as e:
            sys.stderr.write(f"Exception {e} in sequence {sequence}\n")
            sequence = self.clean_seq_encode(sequence)

        sequence = self.truncate(sequence)
        #Hasta ahi, el mask en el get_item,
        sequence, ground_truth = self.mask(sequence)
        return sequence, ground_truth

class ProteinDataset(object):
    def __init__(self, file_path, max_seq_length=1024, masking_rate=0.15):
        self.max_seq_length = max_seq_length
        self.sequence_cache, self.sequence_sizes = self.read_file(file_path)
        self.tokenizer = ProteinTokenizer(self.max_seq_length, masking_rate)

    def read_file(self, file_path):
        if file_path.endswith('.fasta') or file_path.endswith('.fa'):
            sequence_strs, sequence_sizes = [], []
            with open(file_path, 'r') as fasta_file:
                for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                    sequence = str(record.seq)
                    length = len(sequence) + 2  # Add 2 to the length
                    if length > self.max_seq_length:
                        length = self.max_seq_length  # Cap the length at max_seq_length
                    sequence_strs.append(sequence)
                    sequence_sizes.append((length, i))
            return sequence_strs, sequence_sizes
        else:
            
            raise ValueError("Unsupported file format. Only .tar.gz and .fasta are supported.")
    
    def __len__(self):
        return len(self.sequence_cache)

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        sequence = self.sequence_cache[index]
        return self.tokenizer.encode(sequence)
    
class ProteinTokenizerESM1(object):
    def __init__(self, max_seq_length=1024, masking_rate=0.15):
        self.max_seq_length = max_seq_length
        self.masking_rate = masking_rate
        self.base2index = dict((base, idx) for idx, base in enumerate(TOKENS_ESM1))
        self.num_tokens = len(set(self.base2index.values()))
        self.aa_tokens = [self.base2index[token] for token in TOKENS_ESM1[:24]]

    def truncate(self, sequence):
        if sequence.size(dim=0) <= self.max_seq_length:
            return sequence
        j = randint(0, sequence.size(dim=0) - self.max_seq_length)
        return sequence[j:j+self.max_seq_length]

    def mask(self, sequence):
        ground_truth = torch.empty_like(sequence)
        ground_truth.fill_(self.base2index['#'])
        mask = torch.rand(sequence.size()) < self.masking_rate
        mask = mask & (sequence != self.base2index['+']) & (sequence != self.base2index['<']) & (sequence != self.base2index['>'])
        mask_idx = mask.nonzero()
        for i in mask_idx:
            ground_truth[i] = sequence[i]
            p = np.random.rand()
            if p < 0.1:
                sequence[i] = self.aa_tokens[np.random.randint(0, len(self.aa_tokens))]
            elif p < 0.8:
                sequence[i] = self.base2index['#']
        return sequence, ground_truth

    def encode_sequence(self, sequence, sos_eos_tokens=True):
        if sos_eos_tokens:
            encoded_sequence = [self.base2index['<']] + [self.base2index[aa] for aa in sequence] + [self.base2index['>']]
        else:
            encoded_sequence = [self.base2index[aa] for aa in sequence]
        return torch.tensor(encoded_sequence, dtype=torch.long)
    
    def clean_seq_encode(self, sequence, sos_eos_tokens=True):
        clean_sequence = sequence.replace("-","")
        clean_sequence2 = clean_sequence.replace("J","I")
        if sos_eos_tokens:
            clean_sequence = sequence.replace("-","")
            clean_sequence2 = clean_sequence.replace("J","I")
            encoded_sequence = [self.base2index['<']] + [self.base2index[aa] for aa in clean_sequence2] + [self.base2index['>']]
        else:
            encoded_sequence = [self.base2index[aa] for aa in clean_sequence2]
        return torch.tensor(encoded_sequence, dtype=torch.long)
    
    def encode(self, sequence):
        try:
            sequence = self.encode_sequence(sequence)
        except KeyError as e:
            sys.stderr.write(f"Exception {e} in sequence {sequence}\n")
            sequence = self.clean_seq_encode(sequence)

        sequence = self.truncate(sequence)
        #Hasta ahi, el mask en el get_item,
        sequence, ground_truth = self.mask(sequence)
        return sequence, ground_truth
    
    def groupby(self,sequences,truths):
        sep_token =  torch.tensor(self.base2index['.'], dtype=torch.long).unsqueeze(0)
        tokens = 0
        groups = []
        concatenate = []
        concatenate_truths = []
        if not sequences or not truths:
            raise ValueError("Input sequences and truths cannot be empty.")
        if len(sequences) != len(truths):
            raise ValueError("Sequences and truths must have the same length.")
        #Group them with separator to reduce padding to the max on smaller sequences
        for seq,tru in zip(sequences,truths):
            
            size = seq.size(0)
            tokens+= size

            if tokens > self.max_seq_length+2:
                tok_sep_extra = concatenate.pop()
                _ = concatenate_truths.pop()
                assert tok_sep_extra == sep_token

                new_sequence = torch.cat(concatenate,dim=0)
                new_truth = torch.cat(concatenate_truths,dim=0)

                groups.append((new_sequence,new_truth))
                concatenate = []
                concatenate.extend([seq,sep_token])
                concatenate_truths = []
                concatenate_truths.extend([tru,sep_token])
                tokens = 0
                tokens+= size
                #Off by in last iteration, check!!!!!!!!
            else:
                concatenate.extend([seq,sep_token])
                concatenate_truths.extend([tru,sep_token])
            tokens += 1
        #Last iteration, check that concatenate is empty, if not, add an extra group
        if concatenate:
            tok_sep_extra = concatenate.pop()
            _ = concatenate_truths.pop()
            assert tok_sep_extra == sep_token

            new_sequence = torch.cat(concatenate,dim=0)
            new_truth = torch.cat(concatenate_truths,dim=0)

            groups.append((new_sequence,new_truth))

        return groups

    def batch_encode(self, sequences):
        encodings = []
        ground_truths = []

        for sequence in sequences:

            try:
                sequence = self.encode_sequence(sequence)
            except KeyError as e:
                sys.stderr.write(f"Exception {e} in sequence {sequence}\n")
                sequence = self.clean_seq_encode(sequence)

            sequence = self.truncate(sequence)
            sequence, ground_truth = self.mask(sequence)
            encodings.append(sequence)
            ground_truths.append(ground_truth)
        

        #Hasta ahi, el mask en el get_item,
        lot = self.groupby(encodings,ground_truths)
        
        
        return lot
    


class ProteinDatasetESM1(object):
    def __init__(self, file_path, max_seq_length=1024, masking_rate=0.15):
        self.max_seq_length = max_seq_length
        self.sequence_cache, self.sequence_sizes = self.read_file(file_path)
        self.tokenizer = ProteinTokenizerESM1(self.max_seq_length, masking_rate)

    def read_file(self, file_path):
        if file_path.endswith('.fasta') or file_path.endswith('.fa'):
            sequence_strs, sequence_sizes = [], []
            with open(file_path, 'r') as fasta_file:
                for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                    sequence = str(record.seq)
                    length = len(sequence) + 2  # Add 2 to the length
                    if length > self.max_seq_length:
                        length = self.max_seq_length  # Cap the length at max_seq_length
                    sequence_strs.append(sequence)
                    sequence_sizes.append((length, i))
            return sequence_strs, sequence_sizes
        else:
            
            raise ValueError("Unsupported file format. Only .tar.gz and .fasta are supported.")
    
    def __len__(self):
        return len(self.sequence_cache)

    def __getitem__(self, index):
        if isinstance(index, list):
            
            sequences = [self.sequence_cache[x] for x in index]
            #sequences = self.sequence_cache[index[0]:index[-1]]
            output = self.tokenizer.batch_encode(sequences)

            return output
        
        sequence = self.sequence_cache[index]
        return self.tokenizer.encode(sequence)


class ProteinTokenizerNLP(ProteinTokenizerESM1):
    """
    Full NLP mode, everything is concatenated to get 0 padding. 
    We dont give a f about biological batching here, beware before you enter this
    """
    def __init__(self, max_seq_length=1024, masking_rate=0.15):
        super().__init__(max_seq_length, masking_rate)

    def groupby(self,sequences,truths):
        #Need a buffer for carryover sequences
        sep_token =  torch.tensor(self.base2index['.'], dtype=torch.long).unsqueeze(0)
        tokens = 0
        groups = []
        concatenate = []
        concatenate_truths = []
        if not sequences or not truths:
            raise ValueError("Input sequences and truths cannot be empty.")
        if len(sequences) != len(truths):
            raise ValueError("Sequences and truths must have the same length.")
        #Group them with separator to reduce padding to the max on smaller sequences
        for seq,tru in zip(sequences,truths):
            
            size = seq.size(0)
            tokens+= size

            if tokens > self.max_seq_length:

                diff = tokens-self.max_seq_length

                old_seq = seq[-diff:]
                new_seq = seq[:-diff]
                old_tru = tru[-diff:]
                new_tru = tru[:-diff]
                #tok_sep_extra = concatenate.pop()
                #_ = concatenate_truths.pop()
                #assert tok_sep_extra == sep_token
                concatenate.append(old_seq)
                concatenate_truths.append(old_tru)
                new_sequence = torch.cat(concatenate,dim=0)
                new_truth = torch.cat(concatenate_truths,dim=0)

                groups.append((new_sequence,new_truth))
                concatenate = []
                concatenate.extend([new_seq,sep_token])
                concatenate_truths = []
                concatenate_truths.extend([new_tru,sep_token])
                tokens = 0
                tokens+= new_seq.size(0)
                #Off by in last iteration, check!!!!!!!!
            else:
                concatenate.extend([seq,sep_token])
                concatenate_truths.extend([tru,sep_token])
            tokens += 1
        #Last iteration, check that concatenate is empty, if not, add an extra group
        if concatenate:
            tok_sep_extra = concatenate.pop()
            _ = concatenate_truths.pop()
            assert tok_sep_extra == sep_token

            new_sequence = torch.cat(concatenate,dim=0)
            new_truth = torch.cat(concatenate_truths,dim=0)

            groups.append((new_sequence,new_truth))

        return groups
        
class ProteinDatasetNLP(ProteinDatasetESM1):
    """
    Fuck biology, we go full NLP
    """
    def __init__(self, file_path, max_seq_length=1024, masking_rate=0.15):
        super().__init__(file_path, max_seq_length, masking_rate)
        self.tokenizer = ProteinTokenizerNLP(self.max_seq_length, masking_rate)




class ProteinDiskDataset(object):
    def __init__(self, file_path, max_seq_length=1024, masking_rate=0.15,max_tokens=10):
        self.max_seq_length = max_seq_length
        self.max_tokens = max_tokens
        self.current_file_index = 0
        self.counter_sequences = 0
        self.file_paths = self.get_file_paths(file_path)
        self.file_paths.sort()
        self.num_of_files = len(file_path)
        self.sequence_cache = self.read_file(self.file_paths[self.current_file_index])
        self.sequence_sizes = self.get_all_sequence_sizes()
        
        self.tokenizer = ProteinTokenizer(self.max_seq_length, masking_rate)

    def read_file(self, file_path):
        if file_path.endswith('.fasta') or file_path.endswith('.fa'):
            sequence_strs, sequence_sizes = [], []
            with open(file_path, 'r') as fasta_file:
                for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                    sequence = str(record.seq)
                    length = len(sequence) + 2  # Add 2 to the length
                    if length > self.max_seq_length:
                        length = self.max_seq_length  # Cap the length at max_seq_length
                    sequence_strs.append(sequence)
                    sequence_sizes.append((length, i))
            return sequence_strs
        else:
            raise ValueError("Unsupported file format. Only .tar.gz and .fasta are supported.")
    
    def get_file_paths(self, data_dir):
        file_paths = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.fasta') or filename.endswith('.fa'):
                file_paths.append(os.path.join(data_dir, filename))
        return file_paths
    
    def get_all_sequence_sizes(self):
        sequence_sizes = []
        counter = 0
        for file_path in self.file_paths:
            with open(file_path, 'r') as fasta_file:
                for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                    sequence = str(record.seq)
                    length = len(sequence) + 2  # Add 2 to the length
                    if length > self.max_seq_length:
                        length = self.max_seq_length  # Cap the length at max_seq_length
                    sequence_sizes.append((length, counter))
                    counter += 1
        return sequence_sizes

    def length_current_segment(self):
        return len(self.sequence_cache)
    


    def __len__(self):
        return len(self.sequence_sizes)

    def __getitem__(self, index):

        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        #Need to keep summ count, not division
        
        new_index = index - self.counter_sequences
        
        if new_index >= self.length_current_segment():
            
            self.current_file_index += 1
            self.counter_sequences += self.length_current_segment()
            
            
            self.sequence_cache = self.read_file(self.file_paths[self.current_file_index])
            
            
            new_index = new_index - self.length_current_segment()
        
        sequence = self.sequence_cache[new_index]
        return self.tokenizer.encode(sequence)

class ProteinMemMapDataset(object):
    def __init__(self, file_path, max_seq_length=1024, masking_rate=0.15, max_tokens=10):
        self.max_seq_length = max_seq_length
        self.file_path = file_path
        self.masking_rate = masking_rate
        self.max_tokens = max_tokens
        self.storage_dtype = np.uint8
        self.lengths_dtype = np.int32
        self.memmap_path = file_path + "/dataset.mmap"
        self.lengths_path = file_path + "/lengths.mmap"
        self.batches_path = file_path + "/batches.mmap"
        self.pad_integer_uint8 = 255

        if os.path.exists(self.memmap_path):
            print(self.memmap_path)
            print("It exists")
            self.read_mmaps()

        else:
            print("Memmap files not found. Creating new ones.")
            self.create_mmaps()
            print("Created")
            exit(0)
            self.read_mmaps()
        print("DA FAK YOU DOING")
        
    def read_mmaps(self):
        self.data = np.memmap(
                self.memmap_path,
                dtype=self.storage_dtype,
                mode='r',
                shape=(None, self.max_seq_length)
            )

        self.lengths = np.memmap(
                self.lengths_path,
                dtype=self.lengths_dtype,
                mode='r',
                shape=(self.data.shape[0],)
            )
        #TEMP, check this
        self.batches = np.memmap(
                self.batches_path,
                dtype=self.lengths_dtype,
                mode='r',
                shape=(self.data.shape[0],)
            )
        

    def create_wpointers_mmaps(self):
        #Create the pointers for writing
        self.data = np.memmap(
                self.memmap_path,
                dtype=self.storage_dtype,
                mode='w+',
                shape=(self.total_sequences, self.max_seq_length)
            )
        self.lengths = np.memmap(
                self.lengths_path,
                dtype=self.lengths_dtype,
                mode='w+',
                shape=(self.total_sequences,)
            )
    
        self.batches = np.memmap(
                self.batches_path,
                dtype=self.lengths_dtype,
                mode='w+',
                shape=(self.total_sequences,)
            )
        print("created")

    
    def get_paths(self):
        self.file_paths = self.get_file_paths(self.file_path)
        self.file_paths.sort()
        self.num_of_files = len(self.file_path)


    def get_length_dataset_and_sizes(self):
        self.sequence_sizes = self.get_all_sequence_sizes()
        self.total_sequences = len(self.sequence_sizes)


    def determine_optimal_batches(self):
        #Logic of token batch sampler
        sizes = self.sequence_sizes
        print(sizes)
        
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)
        _flush_current_buf()
        #Now this is a lists of lists with the arrays, flatten it, to batch by 
        self.optimal_batches = np.array([i for i, sublist in enumerate(batches) for _ in sublist])

    def create_mmaps(self):
        
        self.get_paths()
        self.get_length_dataset_and_sizes()
        self.determine_optimal_batches()
        print(self.optimal_batches)
        

        self.create_wpointers_mmaps()
        exit(0)
        counter_of_sequences = 0
        for file in self.file_paths:
            sequences = self.read_file(file)
            print(sequences)
            exit(0)
            pass
        print(self.file_paths,self.num_of_files)
        exit(0)
        self.sequence_cache = self.read_file(self.file_paths[self.current_file_index])
        self.sequence_sizes = self.get_all_sequence_sizes()
        self.tokenizer = ProteinTokenizer(self.max_seq_length, self.masking_rate)
        self.process_dataset()



    def read_file(self, file_path):
        if file_path.endswith('.fasta') or file_path.endswith('.fa'):
            sequence_strs, sequence_sizes = [], []
            with open(file_path, 'r') as fasta_file:
                for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                    sequence = str(record.seq)
                    length = len(sequence) + 2  # Add 2 to the length
                    if length > self.max_seq_length:
                        length = self.max_seq_length  # Cap the length at max_seq_length
                    sequence_strs.append(sequence)
                    sequence_sizes.append((length, i))
            return sequence_strs
        else:
            raise ValueError("Unsupported file format. Only .tar.gz and .fasta are supported.")
    
    def get_file_paths(self, data_dir):
        file_paths = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.fasta') or filename.endswith('.fa'):
                file_paths.append(os.path.join(data_dir, filename))
        return file_paths
    
    def get_all_sequence_sizes(self):
        sequence_sizes = []
        counter = 0
        for file_path in self.file_paths:
            with open(file_path, 'r') as fasta_file:
                for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
                    sequence = str(record.seq)
                    length = len(sequence) + 2  # Add 2 to the length
                    if length > self.max_seq_length:
                        length = self.max_seq_length  # Cap the length at max_seq_length
                    sequence_sizes.append((length, counter))
                    counter += 1
        return sequence_sizes

    def length_current_segment(self):
        return len(self.sequence_cache)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Load the sequence length for the given index
        if isinstance(idx, list):
            #testing
            length = self.lengths[idx[0]]
            sequence = torch.from_numpy(self.data[idx[0]:idx[-1], :length].copy())
        
            return sequence
            #return [self.__getitem__(i) for i in idx]
        

        length = self.lengths[idx]
        
        # Load the token sequence from the memmap file
        sequence = torch.from_numpy(self.data[idx, :length].copy())
        
        return sequence
    


    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(idx, list):

            return torch.from_numpy(self.data[idx[0]:[idx[-1]],...])
        # Load only the requested sample from disk
        return torch.from_numpy(self.data[idx])


class TokenBatchSampler:
    def __init__(self, dataset, start_batch=0, max_tokens=10):
        self.start_batch = start_batch
        self.max_tokens = max_tokens
        self.batches = []
        self._prepare_dataset(dataset)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _prepare_dataset(self, dataset):
        sizes = dataset.sequence_sizes
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()

        self.batches = batches[self.start_batch:]
    
class TokenBatchSamplerStL:
    def __init__(self, dataset, start_batch=0, max_tokens=10):
        self.start_batch = start_batch
        self.max_tokens = max_tokens
        self.batches = []
        self._prepare_dataset(dataset)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _prepare_dataset(self, dataset):
        sizes = dataset.sequence_sizes
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()

        self.batches = batches[self.start_batch:]
        
class TokenBatchDiskSampler:
    def __init__(self, dataset, start_batch=0, max_tokens=10):
        self.dataset = dataset
        self.start_batch = start_batch
        self.max_tokens = max_tokens

    def __iter__(self):
        sizes = self.dataset.sequence_sizes
        buf = []
        max_len = 0
        batch_count = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if buf:
                yield buf
            buf = []
            max_len = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                if batch_count >= self.start_batch:
                    yield from _flush_current_buf()
                batch_count += 1
            max_len = max(max_len, sz)
            buf.append(i)

        # Flush remaining buffer if there are any leftover tokens
        if buf and batch_count >= self.start_batch:
            yield buf

    def __len__(self):
        # Estimate length by counting batches without storing them
        sizes = self.dataset.sequence_sizes
        buf = []
        max_len = 0
        batch_count = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                if batch_count >= self.start_batch:
                    batch_count += 1
                buf = []
                max_len = 0
            max_len = max(max_len, sz)
            buf.append(i)

        # Count the last batch if any tokens are left
        if buf:
            batch_count += 1

        return max(0, batch_count - self.start_batch)

class DebugTokenBatchDiskSampler:
    def __init__(self, dataset, start_batch=0, max_tokens=10):
        self.dataset = dataset
        self.start_batch = start_batch
        self.max_tokens = max_tokens

    def __iter__(self):
        sizes = self.dataset.sequence_sizes
        buf = []
        max_len = 0
        batch_count = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if buf:  # Only flush if buf has items
                print(f"Flushing buffer: {buf}")  # Debug statement
                yield buf
            buf = []
            max_len = 0

        for sz, i in sizes:
            print(f"Processing size {sz}, index {i}")  # Debug statement
            # Check if adding this item would exceed the max_tokens
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                if batch_count >= self.start_batch:
                    print("Buffer full, flushing...")  # Debug statement
                    yield from _flush_current_buf()
                batch_count += 1
            max_len = max(max_len, sz)
            buf.append(i)

        # Flush remaining buffer if there are any leftover tokens
        if buf and batch_count >= self.start_batch:
            yield buf

    def __len__(self):
        sizes = self.dataset.sequence_sizes
        buf = []
        max_len = 0
        batch_count = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                if batch_count >= self.start_batch:
                    batch_count += 1
                buf = []
                max_len = 0
            max_len = max(max_len, sz)
            buf.append(i)

        if buf:
            batch_count += 1

        return max(0, batch_count - self.start_batch)
    

class RandomTokenBatchSampler:
    def __init__(self, dataset, start_batch=0, max_tokens=10):
        self.max_tokens = max_tokens
        self.batches = []
        self._prepare_dataset(dataset)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _prepare_dataset(self, dataset):
        sizes = dataset.sequence_sizes
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > self.max_tokens:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()

        self.batches = batches
        np.random.shuffle(self.batches)




class MemMapBatchSampler:
    def __init__(self, dataset, start_batch=0,max_tokens=10):
        self.max_tokens = max_tokens
        self.batches = dataset.batches[start_batch:]
        self.start_batch = start_batch

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def PadBatch(batch):
    '''
    Pads a batch of sequences to the length of the longest sequence in the batch.
    Each batch is a list of lists [(sequence, ground_truth)], where sequence and ground_truth are tensors.
    Note that the ground_truth tensor is also padded with the index of the mask token.
    '''

    if len(batch) == 1:

        batch = batch[0]
    batch_size = len(batch)

    max_seq_len = max([len(seq[0]) for seq in batch])
    padded_sequences = torch.empty(batch_size, max_seq_len, dtype=torch.long)
    padded_sequences.fill_(TOKENS.index('+'))
    padded_ground_truth = torch.empty_like(padded_sequences)
    padded_ground_truth.fill_(TOKENS.index('#'))
    for i, seq in enumerate(batch):
        seq_len = len(seq[0])
        padded_sequences[i, :seq_len] = seq[0]
        padded_ground_truth[i, :seq_len] = seq[1]
    return padded_sequences, padded_ground_truth


