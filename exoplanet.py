import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split

def load_from_csv_folder(folder, seq_len=2048):
    """
    Expect CSVs with columns: time, flux, flux_err (optional), label (0/1 optional)
    Each CSV is one light curve; we'll resample/truncate/pad to seq_len.
    """
    X = []
    y = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(folder, fname))
        # simple canonicalization - assume flux column exists
        if 'flux' not in df.columns:
            raise ValueError(f"{fname} missing 'flux' column")
        flux = df['flux'].values.astype(np.float32)
        flux = _pad_or_trim(flux, seq_len)
        X.append(flux)
        label = 0
        if 'label' in df.columns:
            label = int(df['label'].iloc[0])
        y.append(label)
    X = np.array(X)[..., np.newaxis]  # shape (N, T, 1)
    y = np.array(y)
    return X, y


def load_from_lightkurve(target, mission='TESS', sector=None, seq_len=2048, cadence='long'):
   
    try:
        from lightkurve import search_lightcurve
    except Exception as e:
        raise RuntimeError("lightkurve not installed. pip install lightkurve") from e

    
    lc_search = search_lightcurve(target, mission=mission)
    lc_collection = lc_search.download_all()
    
    lc = lc_collection.stitch() if hasattr(lc_collection, 'stitch') else lc_collection[0]
    flux = lc.flux.value.astype(np.float32)
    flux = np.nan_to_num(flux, nan=np.nanmedian(flux))
    flux = _pad_or_trim(flux, seq_len)
    X = flux[np.newaxis, ..., np.newaxis]
    y = np.array([0], dtype=np.int32)
    return X, y


def _pad_or_trim(arr, length):
    """center-pad or trim a 1D array to length"""
    if len(arr) == length:
        return arr
    elif len(arr) > length:
        start = (len(arr) - length) // 2
        return arr[start:start+length]
    else:
        pad_len = length - len(arr)
        left = pad_len // 2
        right = pad_len - left
        pad_val = np.nanmedian(arr) if len(arr)>0 else 0.0
        return np.pad(arr, (left, right), 'constant', constant_values=(pad_val, pad_val))

def normalize_flux(X):
    """
    Normalize per-sample: (flux - median) / std
    X shape: (N, T, 1)
    """
    X_norm = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        seq = X[i, :, 0]
        med = np.median(seq)
        std = np.std(seq - med) + 1e-9
        X_norm[i, :, 0] = (seq - med) / std
    return X_norm

def detrend_placeholder(X):

    return X


def phase_fold_placeholder(X, period, time_array=None):
    raise NotImplementedError("Implement folding when you have time & period info.")



def build_cnn_model(input_length=2048, channels=1, dropout_rate=0.3):
    inp = layers.Input(shape=(input_length, channels))
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', activation='relu')(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model



def train_model(model, X_train, y_train, X_val, y_val, out_dir='models', epochs=20, batch_size=32):
    os.makedirs(out_dir, exist_ok=True)
    cb = [
        callbacks.ModelCheckpoint(os.path.join(out_dir, 'best_model.h5'),
                                  save_best_only=True, monitor='val_auc', mode='max'),
        callbacks.EarlyStopping(monitor='val_auc', patience=6, mode='max', restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=cb)
    return history

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, return_dict=True)
    print("Test results:", results)
    return results


def make_synthetic_dataset(n_samples=1024, seq_len=2048, transit_depth=0.01, transit_width=10, random_seed=0):
    """
    rng = np.random.RandomState(random_seed)
    X = np.zeros((n_samples, seq_len), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int32)
    for i in range(n_samples):
        base = 1.0 + 0.001 * rng.randn(seq_len)  # small noise baseline
        if rng.rand() < 0.5:
            # inject transit: single box at random position
            pos = rng.randint(seq_len - transit_width)
            depth = transit_depth * (0.5 + rng.rand())
            base[pos:pos+transit_width] -= depth
            y[i] = 1
        X[i, :] = base
    X = X[..., np.newaxis]
    return X, y


def main_smoke_test():
    print("Building synthetic dataset...")
    X, y = make_synthetic_dataset(n_samples=512, seq_len=2048)
    X = detrend_placeholder(X)
    X = normalize_flux(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("Building model...")
    model = build_cnn_model(input_length=X.shape[1], channels=1)
    model.summary()

    print("Training (this is a quick smoke test, reduce epochs for fast runs)...")
    hist = train_model(model, X_train, y_train, X_val, y_val, epochs=6, batch_size=32)

    print("Evaluating...")
    evaluate_model(model, X_test, y_test)

    # save final model
    model.save('models/final_model.h5')
    print("Saved model to models/final_model.h5")

normalize = normalize_flux
pad_or_trim = _pad_or_trim

if __name__ == "__main__":
    main_smoke_test()



