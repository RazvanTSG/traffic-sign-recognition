import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

LABEL_MAP = {
    0: "Speed 20", 1: "Speed 30", 2: "Speed 50", 3: "Speed 60", 4: "Speed 70",
    5: "Speed 80", 6: "End 80", 7: "Speed 100", 8: "Speed 120", 9: "No passing",
    10: "No passing >3.5t", 11: "Right-of-way next", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Trucks prohibited", 17: "No entry",
    18: "General caution", 19: "Curve left", 20: "Curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery", 24: "Narrow right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children", 29: "Bicycles",
    30: "Ice/Snow", 31: "Wild animals", 32: "End limits", 33: "Right ahead",
    34: "Left ahead", 35: "Ahead only", 36: "Straight or Right", 37: "Straight or Left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout", 41: "End no passing",
    42: "End no pass >3.5t"
}

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_input = tf.keras.Input(shape=(30, 30, 3))
    x = grad_input
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    grad_model = tf.keras.models.Model(inputs=grad_input, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def explain_prediction(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    
    last_conv_layer_name = next((layer.name for layer in reversed(model.layers) if 'conv' in layer.name), None)
    if not last_conv_layer_name:
        sys.exit("No convolutional layer found in the model.")

    img = tf.keras.utils.load_img(image_path, target_size=(30, 30))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array[..., ::-1] # BGR to RGB if needed
    img_batch = tf.expand_dims(img_array, 0)

    preds = model.predict(img_batch)
    top_pred_index = np.argmax(preds[0])
    print(f"Prediction: {LABEL_MAP.get(top_pred_index, 'Unknown')} ({100 * np.max(preds):.2f}%)")

    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name, top_pred_index)
    heatmap = np.uint8(255 * heatmap)

    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap).resize((300, 300))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    original_img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(image_path, target_size=(300, 300)))
    superimposed_img = tf.keras.utils.array_to_img(jet_heatmap * 0.4 + original_img)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1), plt.title("Original"), plt.imshow(original_img.astype('uint8')), plt.axis('off')
    plt.subplot(1, 3, 2), plt.title("Attention Heatmap"), plt.imshow(heatmap), plt.axis('off')
    plt.subplot(1, 3, 3), plt.title("Overlay"), plt.imshow(superimposed_img), plt.axis('off')
    plt.tight_layout()
    
    output_filename = "heatmap_analysis.png"
    plt.savefig(output_filename)
    print(f"SUCCESS: Visualization saved to '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python gradcam.py <image.ppm> <model.h5>")
    explain_prediction(sys.argv[1], sys.argv[2])
