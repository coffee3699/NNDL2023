import numpy as np
from keras.saving import load_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


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


def generate_name_with_visualization(model, char_to_idx, idx_to_char, seed, max_len=10):
    seed_sequence = [char_to_idx[char] for char in seed]
    seed_sequence = np.pad(seed_sequence, (0, max_len - len(seed_sequence)), 'constant')

    for _ in range(max_len - len(seed)):
        x_pred = np.zeros((1, max_len))
        x_pred[0, :len(seed_sequence)] = seed_sequence

        preds = model.predict(x_pred, verbose=0)[0]
        top_indices = np.argsort(preds)[-5:]

        next_char = ''
        for idx in top_indices[::-1]:
            if idx_to_char[idx].islower() or idx_to_char[idx] == '|':
                next_char = idx_to_char[idx]
                break

        if next_char == '|' or not next_char:
            break
        seed += next_char
        seed_sequence = np.roll(seed_sequence, -1)
        seed_sequence[-1] = idx

        top_chars = [idx_to_char[idx] if idx_to_char[idx] != '|' else '<EOS>' for idx in top_indices]
        plt.figure(figsize=(10, 4))
        plt.bar(top_chars, preds[top_indices])
        plt.title(f"Top 5 predictions after '{seed[:-1]}'")
        plt.show()

    return seed.replace('|', '')


def main():
    print("Welcome to the RNN Name Generation System!")
    print("Loading pre-trained model...")

    # Load pre-trained model
    model = load_model('SimpleRNN.keras')

    file_path = 'dataset/merged/all_names.txt'
    names = load_data(file_path)
    _, _, char_to_idx = create_dataset(names)
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    print("Pre-trained model loaded.")

    while True:
        seed = input("Enter a seed for name generation (or 'exit' to quit): ").strip()
        if seed.lower() == 'exit':
            break

        while True:
            generated_name = generate_name_with_visualization(model, char_to_idx, idx_to_char, seed)
            print("Generated Name:", generated_name)

            satisfied = input("Are you satisfied with this name? (yes/no): ").strip().lower()
            if satisfied == 'yes':
                break

        continue_generation = input("Would you like to generate another name? (yes/no): ").strip().lower()
        if continue_generation != 'yes':
            break

    print("Thank you for using the RNN Name Generation System!")


if __name__ == "__main__":
    main()
