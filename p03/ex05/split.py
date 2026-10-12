import sys
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.csv>")
        return

    filename = sys.argv[1]

    df = pd.read_csv(filename)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.8)

    training = df.iloc[:split]
    validation = df.iloc[split:]

    training.to_csv("Training_knight.csv", index=False)
    validation.to_csv("Validation_knight.csv", index=False)


if __name__ == "__main__":
    main()