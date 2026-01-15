# pad and unpad attention

在之前的实现中我们的Attention输入往往是(batch, seq_len, hidden_dim) 这样一个三维度的数组，对于这样一个数组而言，其需要对长度不满足seq_len的输入进行pad到seq_len，某种程序上增加了这个冗余计算的量

