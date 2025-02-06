import os

template = """---
datasets:
- {dataset_id}
base_model:
- {model_id}
library_name: transformers, deltazip
---

## {model_id} - {scheme} Compression

This is a compressed model using [deltazip](https://github.com/eth-easl/deltazip).

[Paper](https://arxiv.org/abs/2312.05215), [Compression Tool](https://github.com/eth-easl/deltazip), [Inference Engine (Soon)](https://github.com/eth-easl/deltazip).

## Compression Configuration

- Base Model: {model_id}
- Compression Scheme: {scheme}
- Dataset: {dataset_id}
- Dataset Split: {ds_split}
- Max Sequence Length: {seq_len}
- Number of Samples: {n_samples}

## Sample Output

#### Prompt: 

```
{prompt}
```

#### Output: 

```
{output}
```

## Evaluation

<TODO>

"""

def generate_readme(config) -> str:
    readme = template.format(
        model_id=config['model_id'],
        scheme=config['scheme'],
        dataset_id=config['dataset_id'],
        ds_split=config['ds_split'],
        seq_len=config['seq_len'],
        n_samples=config['n_samples'],
        prompt=config['prompt'],
        output=config['output']
    )
    return readme

def upload_and_delete(org_id, model_id, local_path):
    cmd = f"huggingface-cli upload {org_id}/{model_id} {local_path} --repo-type model"
    os.system(cmd)
    os.system(f"rm -rf {local_path}")

VICUNA_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{{ system_message }}

{% for message in loop_messages %}
  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}
  
  {% if message['role'] == 'user' %}
    {{ 'USER: ' + message['content'].strip() + '\n' }}
  
  {% elif message['role'] == 'assistant' %}
  
    {{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}
  {% endif %}
  
{% endfor %}

{% if add_generation_prompt %}
  {{ 'ASSISTANT:' }}
{% endif %}"""

def update_chat_template(model_name: str):
    if 'vicuna' in model_name.lower():
        return VICUNA_TEMPLATE.strip()
    else:
        return None