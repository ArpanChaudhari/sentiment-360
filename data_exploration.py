import pandas as pd
from datasets import load_dataset
import os


def main():
    print("Downloading SemEval-2014 Task 4 Laptop Reviews dataset")

    try:
        # load laptop dataset from Hugging Face
        ds = load_dataset("tomaarsen/setfit-absa-semeval-laptops")

        # convert to pandas DataFrame for easier Exploration
        train_df = pd.DataFrame(ds["train"])
        test_df = pd.DataFrame(ds["test"])

        # Create a data directory
        os.makedirs("data", exist_ok=True)

        train_df.to_csv("data/laptop_train.csv", index=False)
        test_df.to_csv("data/laptop_test.csv", index=False)

        print(f"Successfully downloaded Train size: {len(train_df)} reviews")
        print(f"Successfully downloaded Test size: {len(test_df)} reviews")
        print(
            "Data Successfully saved to data/laptop_train.csv and data/laptop_test.csv"
        )

        # Display a quick previws of fiest 5 row
        print(train_df.head())
        pass
    except Exception as e:
        print(f"Error downloading dataset:{e}")


if __name__ == "__main__":
    main()
