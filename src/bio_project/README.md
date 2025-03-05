## Structure

- `preprocessing/`: contains the code for preprocessing the data (CLAM)
- `models/`: contains the code for the models
- `utils/`: contains the code for the utility functions
- `features_extraction/`: contains the code for extracting the features
- `inference/`: contains the code for the inference

## Traning

To train the model, setup the `config_train.yaml` file and run the following command:

```bash
python main.py --config config_train.yaml
```

## Inference

For running inference put the slide in the `inference/input_slide` directory, setup the `config_inference.yaml` file and run the following command:

```bash
python inference.py --config config_inference.yaml
```

All the args needs to be passed in the `config_train.yaml` and `config_inference.yaml` files.
