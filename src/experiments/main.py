import logging
import argparse
from pathlib import Path
import torch
import signal
import sys

from config import LOG_CONFIG
from experiment import ExperimentRunner

# Setup logging
logging.basicConfig(
    level=LOG_CONFIG["level"],
    format=LOG_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    print("\nUser aborted training. Cleaning up...")
    sys.exit(0)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run sarcasm detection experiments")
    
    parser.add_argument(
        "--mode",
        choices=["all", "single"],
        default="all",
        help="Run all experiments or single experiment"
    )
    
    parser.add_argument(
        "--stemming",
        choices=["with_stemming", "without_stemming"],
        help="Stemming mode for single experiment"
    )
    
    parser.add_argument(
        "--split",
        choices=["70_30", "80_20"],
        help="Split mode for single experiment"
    )
    
    parser.add_argument(
        "--dimension",
        type=int,
        choices=[512, 768, 1024],
        help="Model dimension for single experiment"
    )
    
    return parser.parse_args()

def main():
    """Main function to run experiments"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner()
        
        if args.mode == "all":
            # Run all experiments
            logger.info("Running all experiments")
            runner.run_all_experiments()
        else:
            # Run single experiment
            if not all([args.stemming, args.split, args.dimension]):
                raise ValueError("For single experiment mode, all parameters must be specified")
            
            logger.info(f"Running single experiment - Stemming: {args.stemming}, "
                       f"Split: {args.split}, Dimension: {args.dimension}")
            
            runner.run_single_experiment(
                stemming_mode=args.stemming,
                split_mode=args.split,
                dimension=args.dimension
            )
        
        # Get and log best experiment
        best_experiment = runner.get_best_experiment()
        logger.info(f"Best experiment: {best_experiment['experiment_name']}")
        logger.info(f"Best accuracy: {best_experiment['test_metrics']['accuracy']:.4f}")
        
        logger.info("All experiments completed successfully")
        
    except KeyboardInterrupt:
        print("\nTraining aborted by user. Cleaning up...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 