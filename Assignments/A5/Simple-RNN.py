import numpy as np
from keras import Sequential
from keras.layers import Embedding, Dense
from keras.layers import SimpleRNN
from matplotlib import pyplot as plt


# Data Preprocessing
def load_data(_file_path):
    with open(_file_path, 'r') as file:
        _names = file.readlines()
    _names = [name.strip() + '\n' for name in _names]  # Add EOS token after each name
    return _names


def create_dataset(_names):
    chars = sorted(list(set(''.join(_names))))
    _char_to_idx = {ch: i for i, ch in enumerate(chars)}
    _sequences = []
    _next_chars = []
    for name in _names:
        for i in range(1, len(name)):
            _sequences.append(name[:i])
            _next_chars.append(name[i])
    return _sequences, _next_chars, _char_to_idx


# Model Definition
def build_model(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=10))
    model.add(SimpleRNN(units=10))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# Modified Training Function with History
def train_model(model, sequences, next_chars, char_to_idx, batch_size=128, epochs=50):
    x = np.zeros((len(sequences), len(max(sequences, key=len))), dtype=np.int32)
    y = np.zeros((len(sequences), len(char_to_idx)), dtype=np.bool_)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t] = char_to_idx[char]
        y[i, char_to_idx[next_chars[i]]] = 1
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs)
    return history


# Visualize Training Loss Curve
def plot_loss_curve(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


# Modified Name Generation Function
def generate_name_with_visualization(model, char_to_idx, idx_to_char, seed, max_len=10):
    for _ in range(max_len):
        x_pred = np.zeros((1, len(seed)))
        for t, char in enumerate(seed):
            x_pred[0, t] = char_to_idx[char]

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = idx_to_char[next_index]
        seed += next_char

        # Visualization with Current Name Prefix
        top_indices = np.argsort(preds)[-5:]
        plt.figure(figsize=(10, 4))
        plt.bar([idx_to_char[idx] for idx in top_indices], preds[top_indices])
        plt.title(f"Top 5 predictions after '{seed[:-1]}'")
        plt.show()

        if next_char == '\n':
            break
    return seed


# Main Execution Block
if __name__ == '__main__':
    file_path = 'dataset/merged/all_names.txt'
    names = load_data(file_path)
    sequences, next_chars, char_to_idx = create_dataset(names)
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    model = build_model(len(char_to_idx))
    history = train_model(model, sequences, next_chars, char_to_idx, epochs=20)

    # Plot Training Loss Curve
    plot_loss_curve(history)

    # Example Usage
    seed = "Alice"
    print(generate_name_with_visualization(model, char_to_idx, idx_to_char, seed))
