import numpy as np
from keras import Sequential
from keras.layers import Embedding, Dense, SimpleRNN
from keras.layers import Masking
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences


# Data Preprocessing
def load_data(_file_path):
    with open(_file_path, 'r') as file:
        _names = file.readlines()
    _names = [name.strip() + '|' for name in _names]  # Use '|' as EOS token
    return _names


def create_dataset(_names):
    chars = sorted(list(set(''.join(_names))))
    _char_to_idx = {ch: i for i, ch in enumerate(chars)}
    _sequences = []
    _next_chars = []
    for name in _names:
        for i in range(1, len(name)):
            _sequences.append([_char_to_idx[char] for char in name[:i]])
            _next_chars.append(_char_to_idx[name[i]])
    # Padding sequences
    _sequences = pad_sequences(_sequences, padding='post')
    return _sequences, np.array(_next_chars), _char_to_idx


def build_model(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=10))
    model.add(Masking(mask_value=0))
    model.add(SimpleRNN(units=100, dropout=0.15))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def train_model(model, sequences, next_chars, vocab_size, batch_size=128, epochs=100):
    y = to_categorical(next_chars, num_classes=vocab_size)
    history = model.fit(sequences, y, batch_size=batch_size, epochs=epochs)
    return history


# Visualize Training Loss Curve
def plot_loss_curve(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


def generate_name_with_visualization(model, char_to_idx, idx_to_char, seed, max_len=15):
    seed_sequence = [char_to_idx[char] for char in seed]
    seed_sequence = np.pad(seed_sequence, (0, max_len - len(seed_sequence)), 'constant')

    for _ in range(max_len - len(seed)):
        x_pred = np.zeros((1, max_len))
        x_pred[0, :len(seed_sequence)] = seed_sequence

        preds = model.predict(x_pred, verbose=0)[0]
        top_indices = np.argsort(preds)[-5:]

        # Find the next most likely lowercase character
        next_char = ''
        for idx in top_indices[::-1]:
            if idx_to_char[idx].islower() or idx_to_char[idx] == '|':
                next_char = idx_to_char[idx]
                break

        if next_char == '|' or not next_char:  # Check for EOS token or no valid lowercase char found
            break
        seed += next_char
        seed_sequence = np.roll(seed_sequence, -1)
        seed_sequence[-1] = idx

        # Visualization with Current Name Prefix
        top_chars = [idx_to_char[idx] if idx_to_char[idx] != '|' else '<EOS>' for idx in top_indices]
        plt.figure(figsize=(10, 4))
        plt.bar(top_chars, preds[top_indices])
        plt.title(f"Top 5 predictions after '{seed[:-1]}'")
        plt.show()

    return seed.replace('|', '')  # Remove EOS token for final output


def see_train_data(sequences, next_chars, num_samples=10):
    print("Displaying top {} training data samples:".format(num_samples))
    for i in range(min(num_samples, len(sequences))):
        print("Sequence: {:15s} Next Char: {}".format(repr(sequences[i]), repr(next_chars[i])))


if __name__ == '__main__':
    file_path = 'dataset/merged/all_names.txt'
    names = load_data(file_path)
    sequences, next_chars, char_to_idx = create_dataset(names)
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    see_train_data(sequences, next_chars, num_samples=10)

    model = build_model(vocab_size=len(char_to_idx))
    history = train_model(model, sequences, next_chars, epochs=100, vocab_size=len(char_to_idx))

    # Plot Training Loss Curve
    plot_loss_curve(history)

    # Example Usage
    seed = "Eli"
    print(generate_name_with_visualization(model, char_to_idx, idx_to_char, seed))

    model.save('SimpleRNN.keras')
