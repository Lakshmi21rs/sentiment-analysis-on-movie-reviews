from src.train_model import train_and_save_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    print("🔁 Training the model...")
    train_and_save_model()

    print("\n📊 Evaluating the model...")
    evaluate_model()
