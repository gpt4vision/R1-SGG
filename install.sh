


pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

#pip install transformers@git+https://github.com/huggingface/transformers.git@2c2495cc7b0e3e2942a9310f61548f40a2bc8425
pip install transformers==4.50.3

pip install trl@git+https://github.com/huggingface/trl.git@ece6738686a8527345532e6fed8b3b1b75f16b16

pip install --upgrade --no-build-isolation flash-attn==2.7.4.post1

# for GH200,
#MAX_JOBS=20 pip install --upgrade --no-build-isolation flash-attn==2.7.4.post1

#git clone https://github.com/triton-lang/triton.git && git checkout 85267600 && cd triton && \
#pip install -r python/requirements.txt # build-time dependencies && \
#pip install -e python

pip install -r requirements.txt

# for GH200,
#pip uninstall -y vllm &&git clone https://github.com/vllm-project/vllm.git&& cd vllm && git checkout ed6e9075d31e32c8548b480\
#python use_existing_torch.py && pip install -r requirements/build.txt && pip install --no-build-isolation -e .

pip install -e .
