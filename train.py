"""
train.py - Plant Disease Detection CNN Training Pipeline

Creates stratified Train / Validation / Test partitions, trains a Convolutional
Neural Network (CNN) on PlantVillage leaf images, evaluates on the unseen test
partition, and exports:
  1. models/plant_model.keras
  2. models/classes.json
  3. models/metrics.json
  4. models/training_history.png
  5. models/confusion_matrix.png
"""

import os
import sys
import json
import shutil
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Ensure Keras runs with the PyTorch backend on Windows for reliable execution
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import layers, models, callbacks

# Project configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "plant_model.keras")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

# Hyperparameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

def check_dataset():
    """Ensure dataset exists and contains images."""
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("[INFO] Dataset not found. Downloading PlantVillage dataset...")
        from download_data import download_dataset
        download_dataset(images_per_class=100)

def prepare_stratified_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits images from dataset/ into deterministic dataset_split/train, val, test
    directories so that all 10 classes are evenly represented in every partition.
    """
    train_dir = os.path.join(SPLIT_DIR, "train")
    val_dir = os.path.join(SPLIT_DIR, "val")
    test_dir = os.path.join(SPLIT_DIR, "test")

    # If split already exists with images, reuse it
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) >= 8:
        print("[INFO] Reusing existing stratified dataset splits in 'dataset_split/'")
        return train_dir, val_dir, test_dir

    print("[INFO] Organizing dataset into stratified Train (70%), Val (15%), Test (15%) splits...")
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    random.seed(RANDOM_SEED)
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    for cls in classes:
        src_cls_dir = os.path.join(DATASET_DIR, cls)
        imgs = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(imgs)

        n_total = len(imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        for subset_name, subset_imgs, target_root in [
            ("train", train_imgs, train_dir),
            ("val", val_imgs, val_dir),
            ("test", test_imgs, test_dir),
        ]:
            dest_dir = os.path.join(target_root, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img_name in subset_imgs:
                shutil.copy2(
                    os.path.join(src_cls_dir, img_name),
                    os.path.join(dest_dir, img_name)
                )

    print("[DONE] Stratified dataset splits created successfully.")
    return train_dir, val_dir, test_dir

def build_cnn_model(num_classes):
    """
    Constructs a high-performance Convolutional Neural Network (CNN)
    optimized for plant disease leaf pathology identification.
    Uses sequential feature extraction blocks with dropout regularization.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        
        # Rescaling pixel values to [0.0, 1.0]
        layers.Rescaling(1.0 / 255.0),
        
        # Data Augmentation (active during training only)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Block 4
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.35),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history):
    """Generates and saves accuracy and loss progression curves."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#2e7d32', linewidth=2.5)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#66bb6a', linewidth=2.5, linestyle='--')
    plt.title('Training & Validation Accuracy', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#c62828', linewidth=2.5)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ef5350', linewidth=2.5, linestyle='--')
    plt.title('Training & Validation Loss', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=180)
    plt.close()
    print(f"[SAVED] Training curves saved to: {HISTORY_PLOT_PATH}")

def plot_confusion_matrix(cm, class_names):
    """Generates a heatmap of the test dataset confusion matrix."""
    plt.figure(figsize=(10, 8))
    clean_labels = [c.replace('___', ' - ').replace('_', ' ') for c in class_names]
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Greens', 
        xticklabels=clean_labels, 
        yticklabels=clean_labels,
        cbar=True,
        linewidths=0.5
    )
    plt.title('Test Dataset Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close()
    print(f"[SAVED] Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

def train():
    print("=" * 70)
    print("   PLANT DISEASE DETECTION - CNN MODEL TRAINING PIPELINE")
    print("=" * 70)
    check_dataset()
    train_dir, val_dir, test_dir = prepare_stratified_splits()

    print("\n[1/5] Loading stratified image datasets...")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nDetected {num_classes} balanced classes:")
    for idx, c in enumerate(class_names):
        print(f"  [{idx}] {c}")
        
    print(f"\nDataset Partitions:")
    print(f"  Training:   {len(train_ds)} batches (~{len(train_ds) * BATCH_SIZE} images)")
    print(f"  Validation: {len(val_ds)} batches (~{len(val_ds) * BATCH_SIZE} images)")
    print(f"  Testing:    {len(test_ds)} batches (~{len(test_ds) * BATCH_SIZE} images)")

    # Save class names mapping
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[SAVED] Class list exported to: {CLASSES_PATH}")

    print("\n[2/5] Building CNN architecture...")
    model = build_cnn_model(num_classes)
    model.summary()

    print("\n[3/5] Training CNN model...")
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )

    print("\n[4/5] Evaluating model on unseen Test Data...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n==================================================")
    print(f"  FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"  FINAL TEST LOSS:     {test_loss:.4f}")
    print(f"==================================================")

    # Collect predictions for Confusion Matrix and Classification Report
    y_true_list = []
    y_pred_list = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        if hasattr(labels, 'numpy'):
            true_labels = labels.numpy()
        else:
            true_labels = np.array(labels)
        y_true_list.extend(true_labels)
        y_pred_list.extend(pred_labels)
        
    y_true = np.array(y_true_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)
    all_class_indices = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=all_class_indices)
    report_dict = classification_report(
        y_true, 
        y_pred, 
        labels=all_class_indices,
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    print("\nClassification Report on Test Data:")
    print(classification_report(y_true, y_pred, labels=all_class_indices, target_names=class_names, zero_division=0))

    # Save visual artifacts
    plot_training_history(history)
    plot_confusion_matrix(cm, class_names)

    # Save metrics JSON for the frontend
    metrics_summary = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "num_classes": num_classes,
        "epochs_trained": len(history.history['accuracy']),
        "final_train_acc": float(history.history['accuracy'][-1]),
        "final_val_acc": float(history.history['val_accuracy'][-1]),
        "history": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']],
        },
        "classification_report": report_dict
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[SAVED] Evaluation metrics saved to: {METRICS_PATH}")

    print("\n[5/5] Saving final trained model as 'models/plant_model.keras'...")
    model.save(MODEL_SAVE_PATH)
    print(f"[SUCCESS] Genuine CNN model successfully saved to: {MODEL_SAVE_PATH}")
    print("\nTraining and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    train()
