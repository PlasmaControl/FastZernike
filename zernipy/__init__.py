"""Set up the device for zernipy."""

import os
import re
import warnings

import numpy as np
from termcolor import colored

config = {"device": None, "avail_mem": None, "kind": None}


def set_device(kind="cpu"):
    """Sets the device to use for computation.

    If kind==``'gpu'``, checks available GPUs and selects the one with the most
    available memory.
    Respects environment variable CUDA_VISIBLE_DEVICES for selecting from multiple
    available GPUs

    Parameters
    ----------
    kind : {``'cpu'``, ``'gpu'``}
        whether to use CPU or GPU.

    """
    config["kind"] = kind
    if kind == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import psutil

        cpu_mem = psutil.virtual_memory().available / 1024**3  # RAM in GB
        config["device"] = "CPU"
        config["avail_mem"] = cpu_mem

    if kind == "gpu":
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import nvgpu

        try:
            devices = nvgpu.gpu_info()
        except FileNotFoundError:
            devices = []
        if len(devices) == 0:
            warnings.warn(colored("No GPU found, falling back to CPU", "yellow"))
            set_device(kind="cpu")
            return

        maxmem = 0
        selected_gpu = None
        gpu_ids = [dev["index"] for dev in devices]
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_ids = [
                s for s in re.findall(r"\b\d+\b", os.environ["CUDA_VISIBLE_DEVICES"])
            ]
            # check that the visible devices actually exist and are gpus
            gpu_ids = [i for i in cuda_ids if i in gpu_ids]
        if len(gpu_ids) == 0:
            # cuda visible devices = '' -> don't use any gpu
            warnings.warn(
                colored(
                    (
                        "CUDA_VISIBLE_DEVICES={} ".format(
                            os.environ["CUDA_VISIBLE_DEVICES"]
                        )
                        + "did not match any physical GPU "
                        + "(id={}), falling back to CPU".format(
                            [dev["index"] for dev in devices]
                        )
                    ),
                    "yellow",
                )
            )
            set_device(kind="cpu")
            return
        devices = [dev for dev in devices if dev["index"] in gpu_ids]
        for dev in devices:
            mem = dev["mem_total"] - dev["mem_used"]
            if mem > maxmem:
                maxmem = mem
                selected_gpu = dev
        config["device"] = selected_gpu["type"] + " (id={})".format(
            selected_gpu["index"]
        )
        config["avail_mem"] = (
            selected_gpu["mem_total"] - selected_gpu["mem_used"]
        ) / 1024  # in GB
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])


if os.environ.get("DESC_BACKEND") == "numpy":
    jnp = np
    use_jax = False
    set_device(kind="cpu")
    print(
        "Using numpy backend, version={}, dtype={}".format(
            np.__version__, np.linspace(0, 1).dtype
        )
    )
else:
    if config.get("device") is None:
        set_device("cpu")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jax
            import jax.numpy as jnp
            import jaxlib
            from jax.config import config as jax_config

            jax_config.update("jax_enable_x64", True)
            if config.get("kind") == "gpu" and len(jax.devices("gpu")) == 0:
                warnings.warn(
                    "JAX failed to detect GPU, are you sure you "
                    + "installed JAX with GPU support?"
                )
                set_device("cpu")
            x = jnp.linspace(0, 5)
            y = jnp.exp(x)
        use_jax = True
        print(
            f"using JAX backend, jax version={jax.__version__}, "
            + f"jaxlib version={jaxlib.__version__}, dtype={y.dtype}"
        )
        del x, y
    except ModuleNotFoundError:
        jnp = np
        x = jnp.linspace(0, 5)
        y = jnp.exp(x)
        use_jax = False
        set_device(kind="cpu")
        warnings.warn(colored("Failed to load JAX", "red"))
        print(
            "Using NumPy backend, version={}, dtype={}".format(np.__version__, y.dtype)
        )
print(
    "Using device: {}, with {:.2f} GB available memory".format(
        config.get("device"), config.get("avail_mem")
    )
)
