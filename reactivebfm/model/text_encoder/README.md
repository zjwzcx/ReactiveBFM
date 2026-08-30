# Text Encoders

Motion planners use one frozen text encoder selected by a single argument:

```bash
--text_encoder bert
--text_encoder t5-base
--text_encoder t5-xl
```

The same argument accepts a full HuggingFace model ID or a local directory.
The loader infers whether it is BERT or T5, freezes it, obtains its output width
from the model config, and builds the planner projection automatically. Text is
truncated to 128 tokens, which is above the motion-caption lengths used here.

The `bert` preset first uses the repo-local runtime model directory
`distilbert-base-uncased/` and otherwise falls back to the HuggingFace model ID
`distilbert/distilbert-base-uncased`. The runtime directory contains model data,
not Python source, and is intentionally ignored by Git and excluded from wheels.

The planner presets `t5-base` and `t5-xl` select
`google/t5-v1_1-base` and `google/t5-v1_1-xl`, respectively. The full
HuggingFace model IDs are also accepted. T5 v1.1 uses the same T5 tokenizer and
encoder interface, so the loader instantiates `T5EncoderModel` and does not load
or run a decoder.

The existing `t5-small`, `t5-large`, and `t5-xxl` presets remain available for
backward compatibility. Other encoder architectures are intentionally rejected
until they are explicitly approved and integrated.

T5 presets first look for a model directory next to this loader, matching the
existing BERT layout. For example, `--text_encoder t5-xl` loads
`reactivebfm/model/text_encoder/t5-xl/` without contacting HuggingFace when that
directory exists. An explicit local directory remains valid as the argument
value.

## Token limits

The BERT and generic T5 planner encoders use the internal
`PLANNER_TEXT_MAX_TOKENS=128` policy and pad only to the longest text in the
current batch.
