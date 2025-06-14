# Synthetic Data Generator

## Supported Option Types

- European (Call & Put)
- American (Call & Put)
- Asian (Call & Put)
- Lookback (Call & Put)
- Knock-Out Barrier (Call & Put)

## Usage

```bash
python main.py
````

This will generate synthetic datasets and save them in the `data/` directory as `.csv` files.

## Project Structure

```
main.py              # Entry point to generate data
src/pricing/         # Pricing logic for different option types
data/                # Output CSV files (not tracked in Git)
```

## Requirements

* Python 3.10+
* NumPy

