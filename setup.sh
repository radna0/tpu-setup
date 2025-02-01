cd ~
export DEBIAN_FRONTEND=noninteractive

curl -f https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo apt-key add -

sudo apt-get update -y
sudo apt-get install software-properties-common -y

DEBIAN_FRONTEND=noninteractive sudo add-apt-repository ppa:deadsnakes/ppa -y


sudo apt install python3.10 python3.10-venv python3.10-dev -y
python3.10 --version

sudo curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10

sudo pip uninstall -y tensorflow tensorflow-cpu

sudo pip install accelerate diffusers transformers loguru peft pandas

pip install --upgrade diffusers

pip install markupsafe==2.0.1

pip install git+https://github.com/huggingface/transformers

sudo apt-get install -y libgl1 libglib2.0-0 google-perftools



# Pytorch XLA
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu --user

pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0.dev+cxx11-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html --user

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]' \
  -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
  -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html \
  --user


# TPU-INFO
python3.10 -m pip install tpu-info


sudo reboot
