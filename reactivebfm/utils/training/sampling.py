import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from reactivebfm.utils.training.common import wrapped_getattr


def get_autoregressive_sample_fn(args, process, gt_prefix=True):
    """Return the per-chunk sampler for the configured generative process."""
    if getattr(args, "model_type", "diffusion") == "flow":
        return process.p_sample_loop_gt_prefix if gt_prefix else process.p_sample_loop
    if gt_prefix:
        return process.p_sample_loop_gt_prefix
    return process.p_sample_loop


class TextConditionCFGSampleModel(nn.Module):
    """Text CFG that always preserves the motion-state prefix.

    The conditional prediction uses both text and state. The CFG baseline masks
    text only, so neither branch creates a text-only/no-state planner input.
    """

    def __init__(self, model):
        super().__init__()
        if model.cond_mask_prob <= 0:
            raise ValueError(
                "Text CFG requires training with cond_mask_prob > 0."
            )
        self.model = model

    def forward(self, x, timesteps, y=None):
        conditioned = self.model(x, timesteps, y)
        state_baseline = dict(y)
        state_baseline["text_uncond"] = True
        without_text = self.model(x, timesteps, state_baseline)
        scale = y["scale"].view(-1, 1, 1, 1)
        return without_text + scale * (conditioned - without_text)

    def __getattr__(self, name, default=None):
        return wrapped_getattr(self, name, default=default)


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn  # p_sample_loop()
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):    # NOTE: sample_fn()
        bs = shape[0]
        n_iterations = (self.required_frames // self.args.pred_len) + 1
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        for _ in range(n_iterations):
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])
            cur_prefix = sample.clone()[..., -self.args.context_len:]  # update

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch


class AutoRegressiveSamplerGTPrefix():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn  # p_sample_loop_gt_prefix() or flow ode_sample_loop()
        self.args = args
        self.required_frames = required_frames

    def sample(self, model, shape, **kargs):
        bs = shape[0]
        
        gt_motion_tensor = kargs['model_kwargs']['y']['gt_motion_tensor']
        gt_prefix = kargs['model_kwargs']['y']['prefix']  # as first prefix
        cur_prefix = gt_prefix.clone()  # as first prefix

        assert gt_prefix is not None and self.args.autoregressive_init == 'data'
        prefix_len = gt_prefix.shape[-1]  # Length of ground truth prefix
        samples_buf = []    # stores all samples
        if self.args.autoregressive_include_prefix:
            samples_buf.append(gt_prefix.clone())   # [bs, njoints, nfeats, context_len] for the first iteration
            samples_buf_torch = torch.cat(samples_buf, dim=-1)
            remaining_frames = self.required_frames - prefix_len    # Calculate how many iterations we need after the prefix
        else:
            remaining_frames = self.required_frames
        n_iterations = (remaining_frames + self.args.pred_len - 1) // self.args.pred_len  # Ceiling division
        
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        if self.args.autoregressive_include_prefix:
            frames_generated = prefix_len   # context_len
        else:
            frames_generated = 0
        print(f"[AutoRegressiveSampler] Starting generation: frames_generated={frames_generated}, required_frames={self.required_frames}, n_iterations={n_iterations}")
        
        # Auto-regressive generation: generate one window at a time
        for iter_idx in range(n_iterations):
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix.clone()
            # # Set first_prefix flag for the first iteration only
            # if iter_idx == 0:
            #     cur_kargs['model_kwargs']['y']['first_prefix'] = True
            # else:
            #     cur_kargs['model_kwargs']['y']['first_prefix'] = False

            # call p_sample_loop_gt_prefix() or flow ode_sample_loop()
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)   # [bs, njoints, nfeats, pred_len]

            assert sample.shape[-1] == self.args.pred_len
            generated_part = sample.clone() # [bs, njoints, nfeats, pred_len]
            samples_buf.append(generated_part)
            samples_buf_torch = torch.cat([samples_buf_torch, generated_part], dim=-1)

            # NOTE: Update prefix for next iteration (free running)
            if self.args.pred_len < self.args.context_len:
                cur_prefix = samples_buf_torch.clone()[..., -self.args.context_len:]
            else:
                # Normal case: pred_len >= context_len, can extract from sample
                cur_prefix = sample.clone()[..., -self.args.context_len:]  # update prefix for next iteration
            
            # # NOTE: always read from gt_motion_tensor (teacher forcing)
            # start_idx = frames_generated - self.args.context_len
            # end_idx = frames_generated
            # assert start_idx >= 0 and end_idx <= gt_motion_tensor.shape[-1]
            # cur_prefix = gt_motion_tensor[..., start_idx:end_idx]
            # print("Teacher forcing!!!")

            frames_generated += self.args.pred_len
            remaining_frames = self.required_frames - frames_generated
            if frames_generated < self.required_frames:
                print(f"[AutoRegressiveSampler] Window {iter_idx+1}/{n_iterations}: total {frames_generated}/{self.required_frames}")
            else:
                print(f"[AutoRegressiveSampler] Window {iter_idx+1}/{n_iterations}: total {self.required_frames}/{self.required_frames}")

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]        
        print(f"[AutoRegressiveSampler] Generated total {full_batch.shape[-1]} frames (required {self.required_frames})")
        return full_batch


class AutoRegressiveSamplerGTPrefixWithEnsemble():
    """
    AutoRegressiveSamplerGTPrefix with Temporal Ensemble functionality.
    
    Prediction frequency: Every pred_len/5 time steps, call the prediction function to generate
    an action_chunk of length pred_len.
    
    Sliding storage: Maintain a queue of size pred_len, storing the latest pred_len action_chunks.
    
    Final action calculation:
    action[t] = W[0] * chunk_0[t] + W[1] * chunk_1[t] + ... + W[pred_len-1] * chunk_{pred_len-1}[t]
    
    Weights: [0.4, 0.2, 0.1, 0.1, 0.1] (most recent to oldest)
    """
    def __init__(self, args, sample_fn, required_frames=196, ensemble_weights=None):
        self.sample_fn = sample_fn  # p_sample_loop_gt_prefix() or flow ode_sample_loop()
        self.args = args
        self.required_frames = required_frames
        # Default weights: [0.4, 0.2, 0.1, 0.1, 0.1] (most recent to oldest)
        if ensemble_weights is None:
            # self.ensemble_weights = [0.4, 0.2, 0.1, 0.1, 0.1]
            self.ensemble_weights = [1.0, 0., 0., 0., 0.]
        else:
            self.ensemble_weights = ensemble_weights
        # Ensure weights match pred_len
        if len(self.ensemble_weights) != self.args.pred_len:
            # If weights don't match, pad or truncate
            if len(self.ensemble_weights) < self.args.pred_len:
                # Pad with last weight
                last_weight = self.ensemble_weights[-1] if self.ensemble_weights else 0.1
                self.ensemble_weights = self.ensemble_weights + [last_weight] * (self.args.pred_len - len(self.ensemble_weights))
            else:
                self.ensemble_weights = self.ensemble_weights[:self.args.pred_len]
        # Normalize weights to sum to 1
        weight_sum = sum(self.ensemble_weights)
        self.ensemble_weights = [w / weight_sum for w in self.ensemble_weights]
        print(f"[AutoRegressiveSamplerWithEnsemble] Initialized with pred_len={self.args.pred_len}, weights={self.ensemble_weights}")

    def sample(self, model, shape, **kargs):
        bs = shape[0]
        
        gt_motion_tensor = kargs['model_kwargs']['y'].get('gt_motion_tensor', None)
        gt_prefix = kargs['model_kwargs']['y']['prefix']

        assert gt_prefix is not None and self.args.autoregressive_init == 'data'
        prefix_len = gt_prefix.shape[-1]  # Length of ground truth prefix
        
        # Calculate prediction frequency: every pred_len/5 time steps
        prediction_step = max(1, self.args.pred_len // 5)  # At least 1 step
        
        # Calculate total number of time steps to generate
        if self.args.autoregressive_include_prefix:
            remaining_frames = self.required_frames - prefix_len
        else:
            remaining_frames = self.required_frames
        
        # Calculate number of predictions needed
        # We predict every prediction_step time steps, and each prediction generates pred_len steps
        # But we only use 1 step from each prediction (or prediction_step steps?)
        # Actually, we need to generate required_frames total, so we need enough predictions
        n_predictions = (remaining_frames + prediction_step - 1) // prediction_step
        
        samples_buf = []
        if self.args.autoregressive_include_prefix:
            samples_buf.append(gt_prefix.clone())   # [bs, njoints, nfeats, context_len] for the first iteration
            frames_generated = prefix_len   # context_len
        else:
            frames_generated = 0
        
        cur_prefix = gt_prefix.clone()
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        print(f"[AutoRegressiveSamplerWithEnsemble] Starting generation: frames_generated={frames_generated}, required_frames={self.required_frames}")
        print(f"[AutoRegressiveSamplerWithEnsemble] Prediction step={prediction_step}, n_predictions={n_predictions}, queue_size={self.args.pred_len}")
        
        # Queue to store last pred_len action_chunks
        # Each chunk is [bs, njoints, nfeats, pred_len]
        chunk_queue = deque(maxlen=self.args.pred_len)
        
        # Final output buffer: stores one action per time step
        final_actions = []
        
        # Generate predictions and compute final actions
        for pred_idx in range(n_predictions):
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix.clone()
            # Set first_prefix flag for the first iteration only
            if pred_idx == 0:
                cur_kargs['model_kwargs']['y']['first_prefix'] = True
            else:
                cur_kargs['model_kwargs']['y']['first_prefix'] = False

            # Call prediction function to generate action_chunk of length pred_len
            action_chunk = self.sample_fn(model, autoregressive_shape, **cur_kargs)   # [bs, njoints, nfeats, pred_len]
            assert action_chunk.shape[-1] == self.args.pred_len
            
            # Add current action_chunk to queue
            chunk_queue.append(action_chunk.clone())
            
            # Compute final actions for the next prediction_step time steps
            # For each time step t in [0, prediction_step-1], compute:
            # action[t] = W[0]*chunk_0[t] + W[1]*chunk_1[t] + ... + W[i]*chunk_i[t] + ...
            # where chunk_i is the i-th chunk in queue (0=most recent, len-1=oldest)
            # Each chunk has pred_len time steps, and we use chunk_i[t] for time step t
            
            for t in range(prediction_step):
                if frames_generated >= self.required_frames:
                    break
                
                # Compute weighted sum: action[t] = W[0]*chunk_0[t] + W[1]*chunk_1[t] + ...
                # chunk_queue[0] is most recent (index 0), chunk_queue[-1] is oldest
                # For time step t, we use chunk_i[t] from each chunk in queue
                action_t = None
                queue_list = list(chunk_queue)  # Convert deque to list for indexing
                
                for i in range(len(queue_list)):
                    # queue_list[i] is the i-th chunk (0=most recent)
                    chunk = queue_list[i]  # [bs, njoints, nfeats, pred_len]
                    # Use weight index i (0 for most recent)
                    if i < len(self.ensemble_weights):
                        weight = self.ensemble_weights[i]
                        # Extract time step t from this chunk
                        # chunk[..., t] gives [bs, njoints, nfeats], we need [bs, njoints, nfeats, 1]
                        chunk_t = chunk[..., t:t+1]  # [bs, njoints, nfeats, 1]
                        if action_t is None:
                            action_t = weight * chunk_t
                        else:
                            action_t = action_t + weight * chunk_t
                
                if action_t is not None:
                    # action_t is [bs, njoints, nfeats, 1]
                    final_actions.append(action_t)
                    frames_generated += 1
                else:
                    # If queue is empty, use the first element of the latest chunk
                    if len(chunk_queue) > 0:
                        action_t = chunk_queue[0][..., t:t+1]
                        final_actions.append(action_t)
                        frames_generated += 1
            
            # Update prefix for next prediction
            # Use the most recent chunk to update prefix
            if len(chunk_queue) > 0:
                latest_chunk = chunk_queue[0]  # Most recent chunk
                if self.args.pred_len < self.args.context_len:
                    # When pred_len < context_len, we need to read from gt_motion_tensor
                    start_idx = frames_generated - self.args.context_len
                    end_idx = frames_generated
                    if start_idx >= 0 and end_idx <= gt_motion_tensor.shape[-1]:
                        cur_prefix = gt_motion_tensor[..., start_idx:end_idx]
                    else:
                        # Fallback: use last context_len frames from latest_chunk
                        cur_prefix = latest_chunk.clone()[..., -self.args.context_len:]
                else:
                    # Normal case: pred_len >= context_len, extract from latest chunk
                    cur_prefix = latest_chunk.clone()[..., -self.args.context_len:]
            
            if frames_generated < self.required_frames:
                print(f"[AutoRegressiveSamplerWithEnsemble] Prediction {pred_idx+1}/{n_predictions}: total {frames_generated}/{self.required_frames} (queue size: {len(chunk_queue)})")
            else:
                print(f"[AutoRegressiveSamplerWithEnsemble] Prediction {pred_idx+1}/{n_predictions}: total {self.required_frames}/{self.required_frames} (queue size: {len(chunk_queue)})")
            
            if frames_generated >= self.required_frames:
                break

        # Concatenate all final actions
        if final_actions:
            # Stack: [n_actions, bs, njoints, nfeats, 1] -> [bs, njoints, nfeats, n_actions]
            final_batch = torch.cat(final_actions, dim=-1)  # [bs, njoints, nfeats, n_actions]
            # Trim to required_frames
            final_batch = final_batch[..., :self.required_frames]
        else:
            # Fallback: if no actions generated, return zeros
            final_batch = torch.zeros((bs, shape[1], shape[2], self.required_frames), 
                                     device=gt_prefix.device, dtype=gt_prefix.dtype)
        
        # Combine with prefix if needed
        if self.args.autoregressive_include_prefix:
            full_batch = torch.cat([gt_prefix.clone(), final_batch], dim=-1)
            full_batch = full_batch[..., :self.required_frames]
        else:
            full_batch = final_batch
        
        print(f"[AutoRegressiveSamplerWithEnsemble] Generated total {full_batch.shape[-1]} frames (required {self.required_frames})")
        return full_batch


class AutoRegressiveSamplerGTPrefixClosedLoop():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn  # p_sample_loop_gt_prefix() or flow ode_sample_loop()
        self.args = args
        self.required_frames = required_frames

    def sample(self, model, shape, **kargs):
        bs = shape[0]
        
        # gt_motion_tensor = kargs['model_kwargs']['y']['gt_motion_tensor']
        gt_prefix = kargs['model_kwargs']['y']['prefix']  # as first prefix
        cur_prefix = gt_prefix.clone()  # as first prefix

        assert gt_prefix is not None and self.args.autoregressive_init == 'data'
        prefix_len = gt_prefix.shape[-1]  # Length of ground truth prefix
        samples_buf = []    # stores all samples
        # if self.args.autoregressive_include_prefix:
        samples_buf.append(gt_prefix.clone())   # [bs, njoints, nfeats, context_len] for the first iteration
        samples_buf_torch = torch.cat(samples_buf, dim=-1)
        remaining_frames = self.required_frames - prefix_len    # Calculate how many iterations we need after the prefix


        # if self.args.autoregressive_include_prefix:
        #     samples_buf.append(gt_prefix.clone())   # [bs, njoints, nfeats, context_len] for the first iteration
        #     remaining_frames = self.required_frames - prefix_len    # Calculate how many iterations we need after the prefix
        # else:
        #     remaining_frames = self.required_frames
        # n_iterations = (remaining_frames + self.args.pred_len - 1) // self.args.pred_len  # Ceiling division
        n_iterations = 1    # TODO: tracking a frame at a time, and then generate the next chunk
        
        
        cur_prefix = gt_prefix.clone()
        
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        if self.args.autoregressive_include_prefix:
            frames_generated = prefix_len   # context_len
        else:
            frames_generated = 0
        # print(f"[AutoRegressiveSampler] Starting generation: frames_generated={frames_generated}, required_frames={self.required_frames}, n_iterations={n_iterations}")
        
        # Auto-regressive generation: generate one window at a time
        for iter_idx in range(n_iterations):
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix.clone()
            # # Set first_prefix flag for the first iteration only
            # if iter_idx == 0:
            #     cur_kargs['model_kwargs']['y']['first_prefix'] = True
            # else:
            #     cur_kargs['model_kwargs']['y']['first_prefix'] = False

            # call p_sample_loop_gt_prefix() or flow ode_sample_loop()
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)   # [bs, njoints, nfeats, pred_len]

            assert sample.shape[-1] == self.args.pred_len
            generated_part = sample.clone() # [bs, njoints, nfeats, pred_len]
            samples_buf.append(generated_part)
            samples_buf_torch = torch.cat([samples_buf_torch, generated_part], dim=-1)

            # NOTE: Update prefix for next iteration (free running)
            if self.args.pred_len < self.args.context_len:
                cur_prefix = samples_buf_torch.clone()[..., -self.args.context_len:]
            else:
                # Normal case: pred_len >= context_len, can extract from sample
                cur_prefix = sample.clone()[..., -self.args.context_len:]  # update prefix for next iteration
            
            # # NOTE: always read from gt_motion_tensor (teacher forcing)
            # start_idx = frames_generated - self.args.context_len
            # end_idx = frames_generated
            # assert start_idx >= 0 and end_idx <= gt_motion_tensor.shape[-1]
            # cur_prefix = gt_motion_tensor[..., start_idx:end_idx]
            # print("Teacher forcing!!!")

            frames_generated += self.args.pred_len
            remaining_frames = self.required_frames - frames_generated
            # remaining_frames = self.required_frames - frames_generated
            # if frames_generated < self.required_frames:
            #     print(f"[AutoRegressiveSampler] Window {iter_idx+1}/{n_iterations}: total {frames_generated}/{self.required_frames}")
            # else:
            #     print(f"[AutoRegressiveSampler] Window {iter_idx+1}/{n_iterations}: total {self.required_frames}/{self.required_frames}")

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]        
        # print(f"[AutoRegressiveSampler] Generated total {full_batch.shape[-1]} frames (required {self.required_frames})")
        return full_batch
