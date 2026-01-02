# src/training/train.py
"""Dispatcher for training different model types."""
import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train hate speech detection models')
    parser.add_argument('--model', choices=['baseline', 'bilstm', 'textcnn', 'hecan'], default='baseline',
                        help='Model type to train')
    parser.add_argument('--sample', type=int, help='Sample size for quick testing')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for deep models')
    args = parser.parse_args()
    
    cmd = [sys.executable, '-m', f'src.training.train_{args.model}']
    if args.sample:
        cmd.extend(['--sample', str(args.sample)])
    if args.model != 'baseline' and args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
