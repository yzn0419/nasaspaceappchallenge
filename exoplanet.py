import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split

# For real data:
import lightkurve as lk



def fetch_lightcurve(target, mission="TESS", seq_len=2048):
    """
    Fetch a stitched lightcurve for the given target using Lightkurve.
    Returns (X, y) where X shape = (1, seq_len, 1), y = [0] placeholder.
    """
    search = lk.search_lightcurve(target, mission=mission)
    if len(search) == 0:
        raise ValueError(f"No lightcurve found for target {target} in mission {mission}")
    lc_collection = search.download_all()  

    try:
        lc = lc_collection.stitch()
    except Exception:
        
        lc = lc_collection[0]
    
    if hasattr(lc, "pdcsap_flux") and lc.pdcsap_flux is not None:
        flux = lc.pdcsap_flux.value
    else:
        flux = lc.flux.value
    
    flux = np.nan_to_num(flux, nan=np.nanmedian(flux))
    
    flux = pad_or_trim(flux, seq_len)
    X = flux[np.newaxis, :, np.newaxis].astype(np.float32)
    y = np.array([0], dtype=np.int32)  # placeholder label
    return X, y, lc.time.value  # return time array too in case you want to fold

def pad_or_trim(arr, length):
    if arr.shape[0] == length:
        return arr
    elif arr.shape[0] > length:
        start = (arr.shape[0] - length) // 2
        return arr[start : start + length]
    else:
        pad_len = length - arr.shape[0]
        left = pad_len // 2
        right = pad_len - left
        med = np.median(arr) if arr.size > 0 else 0.0
        return np.pad(arr, (left, right), "constant", constant_values=(med, med))

def normalize(X):
    """Normalize flux per sample: (flux - median) / stdev"""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        seq = X[i, :, 0]
        med = np.median(seq)
        std = np.std(seq - med) + 1e-9
        X_norm[i, :, 0] = (seq - med) / std
    return X_norm

def detrend_simple(X):
    """Simple detrend placeholder: subtract a rolling median or trend estimate."""
    # Here, do nothing (identity); replace with Savitzky–Golay or spline in real use
    return X

# Optional: fold around period if you know it. Not used yet.

# ----------------------------
# Model definition
# ----------------------------

def build_cnn_model(input_length=2048, channels=1, dropout_rate=0.3):
    inp = layers.Input(shape=(input_length, channels))
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding="same", activation="relu")(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

# ----------------------------
# Training / evaluation
# ----------------------------

def train_model(model, X_train, y_train, X_val, y_val, out_dir="models", epochs=10, batch_size=16):
    os.makedirs(out_dir, exist_ok=True)
    cb = [
        callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best_model.h5"), save_best_only=True, monitor="val_auc", mode="max"
        ),
        callbacks.EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=cb
    )
    return history

def evaluate_model(model, X_test, y_test):
    res = model.evaluate(X_test, y_test, return_dict=True)
    print("Test results:", res)
    return res

# ----------------------------
# Example usage
# ----------------------------

def main():
    # 1. Optionally fetch real data
    target = "TIC 25155310"  # example TESS target (you can swap this)
    print("Fetching real light curve for target:", target)
    try:
        X_real, y_real, time_arr = fetch_lightcurve(target, mission="TESS", seq_len=2048)
        print("Fetched shape:", X_real.shape)
    except Exception as e:
        print("Could not fetch real data:", e)
        X_real, y_real = None, None

    # 2. Synthetic dataset for fallback / mixing
    n_samples = 256
    seq_len = 2048
    X_syn = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
    y_syn = np.zeros((n_samples,), dtype=np.int32)
    rng = np.random.RandomState(0)
    for i in range(n_samples):
        base = 1.0 + 0.001 * rng.randn(seq_len)
        if rng.rand() < 0.5:
            pos = rng.randint(seq_len - 20)
            base[pos: pos + 10] -= 0.01 * (0.5 + rng.rand())
            y_syn[i] = 1
        X_syn[i, :, 0] = base
    X_syn = detrend_simple(X_syn)
    X_syn = normalize(X_syn)

    # 3. Optionally combine real + synthetic or just use synthetic for initial training
    if X_real is not None:
        # You might want to expand real data (e.g. via random cropping, injection)
        X = np.concatenate([X_syn, X_real], axis=0)
        y = np.concatenate([y_syn, y_real], axis=0)
    else:
        X, y = X_syn, y_syn

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 4. Build model
    model = build_cnn_model(input_length=seq_len, channels=1)
    model.summary()

    # 5. Train
    train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=16)

    # 6. Evaluate
    evaluate_model(model, X_test, y_test)

    # 7. Save entire model
    model.save("models/final_model_with_real.h5")
    print("Saved model to models/final_model_with_real.h5")

if __name__ == "__main__":
    main()




