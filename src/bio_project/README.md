## Structure
- `preprocessing/`: contains the code for preprocessing the data (CLAM)
- `models/`: contains the code for the models
- `utils/`: contains the code for the utility functions
- `features_extraction/`: contains the code for extracting the features
- `inference/`: contains the code for the inference


## Traning
To train the model, run the following command:
```bash
python main.py --datasetpath DATASETPATH
```

# Inference

For running inference put the slide in the `inference/input_slide` directory and run the following command:

```bash
python inference.py --config config.yaml
```

All the args needs to be passed in the `config.yaml` file.
