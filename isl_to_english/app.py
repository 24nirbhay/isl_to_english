import argparse
import os
from src.data_collection import collect_data, process_image_dataset
from src.model_train import train_model
from src.real_time_translator import main as run_translator

def main():
    parser = argparse.ArgumentParser(description="ISL to English Translator")
    parser.add_argument("command", choices=["collect", "prepare", "train", "run", "gpu-check"], help="Command to execute")
    parser.add_argument("--gesture", help="Gesture name for data collection")
    args = parser.parse_args()

    # Create necessary directories if they don't exist
    os.makedirs("data/dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if args.command == "collect":
        if not args.gesture:
            print("Please provide a gesture name using --gesture")
            return
        collect_data(args.gesture)
    elif args.command == "prepare":
        # Convert images in data/handsigns into landmark CSVs under data/dataset
        print("Preparing dataset from images in data/handsigns/...")
        process_image_dataset(os.path.join('data', 'handsigns'), os.path.join('data', 'dataset'))
        print("Dataset preparation complete.")
    elif args.command == "train":
        train_model()
    elif args.command == "run":
        run_translator()
    elif args.command == "gpu-check":
        from src.gpu_check import check_gpu
        check_gpu()

if __name__ == "__main__":
    main()
