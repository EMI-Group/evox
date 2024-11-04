# Selecting GPU / CPU

To run your program on a specific GPU, you can use the `CUDA_VISIBLE_DEVICES` environment variable. For example, to run your program on the second GPU, you can use:

```bash
CUDA_VISIBLE_DEVICES=1 python my_program.py
```

To run your program on multiple GPUs, you can use:

```bash
CUDA_VISIBLE_DEVICES=0,1 python my_program.py
```

To disable GPU usage (use CPU), you can use:

```bash
CUDA_VISIBLE_DEVICES="" python my_program.py
```
