###pipeline

---
```python
preprocessing.py -> create_model_inputs.py -> train_.py
```

This directory contains three types of model, cat, lgb and simple feed-forward nn.


### local experiment result

| model  | local cv|
| ------- | ------------- |
| cat  | ~0.655  |
| lgb | - |
| simple nn | ~0.6488 |