# 大模型推理性能评估

推理性能评估主要关注prefill和decoding的时间

- prefill latency：，模型处理prompt的时间,对大模型prompt的全部token进行QKV的计算
- decoding latency：模型处理输入的时间
- e2e latency：指从请求发送到完整接收所有输出 Token 的总耗时

```python
    def evaluate_inference(self, prompts: List[str], max_new_tokens: int = 50, submit_time=""
                           ) -> Dict[str, Any]:

        results = {
            'model_name': self.model_name,
            'timestamp': submit_time if submit_time else datetime.now().strftime('%Y-%m-%d_%H-%M-%S') ,
            'prompt_lengths': [],
            'prefill_latencies': [],
            'decode_latencies': [],
            'generated_texts': [],
            'generated_token_num': [],
            'logits_paths':[] 
        }

        self.gen("Hello, this is a warm-up sequence to prepare the model.", 10, output_logits=False)

        for prompt_id, prompt in enumerate(prompts):
            print("=" * 20 + 'Generating' + "=" * 20)
            print("Prompt : {}".format(prompt))
            ms.runtime.synchronize()
            # prefill阶段耗时
            prefill_start = time.time()
            model_inputs, model_output = self.gen(prompt, 1)
            ms.runtime.synchronize()
            prefill_end = time.time()
            prefill_latency = prefill_end - prefill_start
            print(f"prefill_latency:{prefill_latency}")
            results['prefill_latencies'].append(prefill_latency)
            prompt_length = model_inputs['input_ids'].shape[1]
            results['prompt_lengths'].append(prompt_length)
            ms.runtime.synchronize()
            # prefill+decode阶段耗时
            e2e_latency_start = time.time()
            model_input, model_output = self.gen(prompt, max_new_tokens,output_logits=True)
            ms.runtime.synchronize()
            e2e_latency_end = time.time()
            e2e_latency = e2e_latency_end - e2e_latency_start
            decode_latency = e2e_latency - prefill_latency
            print(f"e2e_latency:{e2e_latency}, decode_latency:{decode_latency}")
            mean_decode_latency = decode_latency / (model_output.sequences.shape[1] - model_input['input_ids'].shape[1])
    
            if model_output.logits is not None:
                # 生成唯一的logits文件名：包含模型名、时间戳、prompt_id
                logits_filename = f"{self.model_name}_{results['timestamp']}_prompt_{prompt_id}.npy"
                logits_filepath = os.path.join(self.output_dir, 'logits', logits_filename)
                os.makedirs(os.path.dirname(logits_filepath), exist_ok=True)

                # 将logits转换为numpy数组并保存为.npy文件
                logits_list = [step_logits.asnumpy() for step_logits in model_output.logits]
                logits_np = np.stack(logits_list, axis=0)  # 按step维度堆叠
                np.save(logits_filepath, logits_np)
                results['logits_paths'].append(logits_filepath)
                
            results['generated_token_num'].append(model_output.sequences.shape[1] - model_input['input_ids'].shape[1])
            results['decode_latencies'].append(mean_decode_latency)
            results['generated_texts'].append(
                self.tokenizer.decode(model_output.sequences[0], skip_special_tokens=True))
            mindspore.runtime.empty_cache()

        results['overall_avg_latencies'] = {
            'avg_prefill_latency': np.mean(results['prefill_latencies']),
            'avg_decode_latency': np.mean(results['decode_latencies'])
        }
        results['memory_allocated'] = mindspore.runtime.max_memory_allocated()/10**9
        results['memory_reserved'] = mindspore.runtime.max_memory_reserved()/10**9
        return results
```