import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x - np.max(x)
        e_x = np.exp(x)
        return e_x / np.sum(e_x)
    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    else:
        raise ValueError("softmax expects 1D or 2D input")

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward inference function ===
def nn_forward_h5(model_arch, weights, data):
    x = data

    for layer in model_arch['config']['layers']:
        class_name = layer['class_name']
        config = layer['config']
        activation = config.get("activation", None)

        if class_name == "Flatten":
            x = flatten(x)

        elif class_name == "Dense":
            layer_name = config["name"]
            W = weights[f"{layer_name}_0"]
            b = weights[f"{layer_name}_1"]
            x = dense(x, W, b)

            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

    return x

def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
