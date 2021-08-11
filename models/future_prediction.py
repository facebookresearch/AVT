"""
Implementation of the future features prediction models.
    Input: (B, C)
    Output: (B, C)
"""
import torch
import torch.nn as nn
import transformers
import logging

import hydra
from common.cluster import KmeansAssigner


class Identity(nn.Module):
    """Wrapper around the Identity fn to drop target_shape etc."""
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, feats, target_shape=None):
        del target_shape  # not needed here
        return feats, feats, {}, {}

    @property
    def output_dim(self):
        return self.in_features


class MLP(nn.Module):
    def __init__(self, in_features, num_layers=2):
        super().__init__()
        self.in_features = in_features
        layers = [[nn.Linear(in_features, in_features),
                   nn.ReLU(inplace=True)] for _ in range(num_layers)]
        # Flatten, remove the last ReLU, and create a sequential
        self.model = nn.Sequential(
            *([item for sublist in layers for item in sublist][:-1]))

    def forward(self, feats, target_shape=None):
        del target_shape
        return feats, self.model(feats), {}, {}

    @property
    def output_dim(self):
        return self.in_features


class T5LM(nn.Module):
    """T5 Language model, for dense future prediction."""
    def __init__(self,
                 in_features: int,
                 output_len: int = 100,
                 avg_last_n: int = -1,
                 inter_dim: int = 256,
                 **kwargs):
        """
        Args:
            output_len: How many outputs to generate
            avg_last_n: From the output_len, take only these many last and
                mean it. This is useful when generting the 1s future clip
                etc. The output will (B, C), without the T dimension
        Return:
            B, T', C where T = output_len if avg_last_n is -1, or B, C
        """
        super().__init__()
        self.reduce_dim = nn.Linear(in_features, inter_dim)
        self.model = transformers.T5ForConditionalGeneration(
            transformers.T5Config(d_model=inter_dim,
                                  vocab_size=inter_dim,
                                  **kwargs))
        self.in_features = in_features
        self.output_len = output_len
        self.avg_last_n = avg_last_n
        self.inter_dim = inter_dim

    def forward(self, feats, target_shape=None):
        """
        Args:
            feats: B, T, C
        """
        del target_shape  # Not using at the moment
        assert feats.ndim == 3, (
            'Must use identity pooling for temporal agg, so temp dim exists')
        # Reduce the dimensionality
        feats = self.reduce_dim(feats)
        # Pad the features for the amount that needs to be generated
        raise NotImplementedError('This is incorrect way.. need to '
                                  'autoregressively generate the future. '
                                  'Do similar to GPT.'
                                  'Also, return past features.')
        padding = torch.zeros(
            (feats.shape[0], self.output_len, feats.shape[-1]),
            dtype=feats.dtype,
            device=feats.device)
        attention_mask = torch.zeros((feats.shape[0], self.output_len),
                                     dtype=torch.bool,
                                     device=feats.device)
        # All padding is to be ignored right...
        # attention_mask[:, :feats.shape[1]] = 1.0
        # Not really sure what are the other objects this is returning
        # TODO figure it out and ideally optimize it to only get logits
        logits, _, _ = self.model(inputs_embeds=feats,
                                  decoder_inputs_embeds=padding,
                                  decoder_attention_mask=attention_mask)
        if self.avg_last_n > 0:
            logits = torch.mean(logits[:, -self.avg_last_n:, :], dim=1)
        return logits, {}, {}, {}

    @property
    def output_dim(self):
        return self.inter_dim


class GPT2LM_v2(nn.Module):
    """GPT2 Language model, for dense future prediction.
    v2: v1 took as input quantized or non quantized features. This one always
    takes unquantized features, and quantizes on the fly if the loss function
    asks for it, by using the embedding matrix. This should allow for the
    model to get actual past features, instead of getting quantized
    """
    def __init__(self,
                 in_features: int,
                 output_len: int = -1,
                 avg_last_n: int = -1,
                 inter_dim: int = 768,
                 future_pred_loss: hydra.types.TargetConf = None,
                 return_past_too: bool = False,
                 **kwargs):
        super().__init__()
        if inter_dim != in_features:
            self.encoder = nn.Linear(in_features, inter_dim, bias=False)
            self.decoder = nn.Linear(inter_dim, in_features, bias=False)
        else:
            self.encoder = nn.Sequential()
            self.decoder = nn.Sequential()
        # This already has the LayerNorm inside residual, as Naman suggested.
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(n_embd=inter_dim,
                                    vocab_size=in_features,
                                    use_cache=True,
                                    **kwargs))
        # Not needed, encoder will take care of it. Keeping it gives the error
        # that there are untrained params in the model. Earlier probably it
        # worked since I was using the LMHead model, which probably tied up
        # weights to the decoder layer, and hence trained it maybe
        del self.gpt_model.wte
        self.output_len = output_len
        self.avg_last_n = avg_last_n
        self.inter_dim = inter_dim
        self.in_features = in_features
        if future_pred_loss is not None:
            self.future_pred_loss = hydra.utils.instantiate(future_pred_loss,
                                                            reduction='none')
        else:
            self.future_pred_loss = None
        self.return_past_too = return_past_too

    def forward(self, feats, target_shape):
        """
        Args:
            feats: tensor of shape (B, T, C)
            target_shape: shape of the output (B, T', n_output)
        """
        if feats.ndim == 2:
            # add back the temporal dimension, which was likely mean pooled
            feats = feats.unsqueeze(1)
        # Decide the output len based on the target_shape
        if len(target_shape) == 3:
            output_len = target_shape[1]
        else:
            output_len = self.output_len
        # The time dimension in already in the middle -> B, T, C
        # That's what huggingface version needs:
        # (batch_size, sequence_length, hidden_size)
        orig_feats = feats
        orig_feats_len = feats.size(1)
        feats = self.encoder(feats)
        past = None
        all_outputs = []
        for _ in range(output_len):
            pred_so_far = sum([el.size(1) for el in all_outputs])
            position_ids = torch.arange(pred_so_far,
                                        pred_so_far + feats.size(1),
                                        dtype=torch.long,
                                        device=feats.device)
            # The past output will encode the previous past AND the new input
            # (you can check the output, it keeps increasing)
            # Got this from
            # https://huggingface.co/transformers/quickstart.html#using-the-past
            # Ideally for the quantized case, I should do some beam
            # search style generation, but perhaps that's for testing...
            outputs = self.gpt_model(inputs_embeds=feats,
                                     past_key_values=past,
                                     position_ids=position_ids)
            last_hidden_state = outputs.last_hidden_state
            past = outputs.past_key_values
            # hidden_states[-1] or last_hidden_state is the embedding from the
            # final layer. Not using logits (earlier was using the LMHead model
            # that returned logits) since that is already decoded to vocab size
            # and I want to have control over the weights of that final matrix
            # Also, the input for the next would be encodings, so need to
            # access the encodings directly
            feats = last_hidden_state[:, -1:, :]
            all_outputs.append(last_hidden_state)
        all_outputs = torch.cat(all_outputs, dim=1)
        # Map back to the original feature dimension
        all_outputs_decoded = self.decoder(all_outputs)
        # Compute a loss on future prediction (teacher forced)
        losses = {}
        if self.future_pred_loss is not None:
            losses = {
                'gpt2_loss':
                self.future_pred_loss(
                    all_outputs_decoded[:, :orig_feats_len - 1],
                    orig_feats[:, 1:orig_feats_len])
            }
        if self.in_features != 1:
            # Want to return encoded feats when in_feats = 1
            all_outputs = all_outputs_decoded
        # Return the actual predictions
        if self.return_past_too:
            # Pad in the GT past (no point using the predicted past when
            # we have the actual past)
            final = torch.cat(
                (orig_feats, all_outputs[:, orig_feats_len - 1:, :]), dim=1)
        elif output_len > 0:
            final = all_outputs[:, -output_len:]
        else:
            final = all_outputs
        if self.avg_last_n > 0:
            final = torch.mean(final[:, -self.avg_last_n:, :], dim=1)
        return all_outputs[:, :orig_feats_len], final, losses, {}

    @property
    def output_dim(self):
        return self.in_features


class GPT2LM(nn.Module):
    """GPT2 Language model, for dense future prediction."""
    def __init__(
            self,
            in_features: int,
            output_len: int = -1,
            output_len_eval: int = -1,  # Same as output_len, used during eval
            avg_last_n: int = -1,
            inter_dim: int = 768,
            future_pred_loss: hydra.types.TargetConf = None,
            return_past_too: bool = False,
            drop_last_n: int = 0,
            quantize_before_rollout: bool = False,
            # This is only relevant when in_features=1 and input is
            # clustered, or if on the fly cluster assgn is requested
            assign_to_centroids: str = None,
            num_cluster_centers: int = 50000,
            freeze_encoder_decoder: bool = False,
            **kwargs):
        super().__init__()
        self.assign_to_centroids = assign_to_centroids
        if self.assign_to_centroids:
            # Since we will be assign the features
            assert in_features != 1
            self.assigner = KmeansAssigner(assign_to_centroids)
            assert self.assigner.num_clusters == num_cluster_centers
        if in_features == 1 or assign_to_centroids:
            # This is a quantized input, so use a embedding matrix
            # NOTE that centroids will not be initialized into the embedding
            # layer here. If you want to do that, use the config options
            # to init this layer using the pth file. Init-ing here would
            # anyway get over-written with the random init later.
            self.encoder = nn.Embedding(num_cluster_centers, inter_dim)
        else:
            self.encoder = nn.Linear(in_features, inter_dim, bias=False)
        # Try out tying the weights of the encoder and decoder as they
        # do in NLP in
        # https://github.com/fairinternal/fairseq-py/blob/66e334580be7379da6ab3c93ef9f018045fc560b/fairseq/models/transformer.py#L636
        # It may not make sense here because their they tie the linear
        # layer with nn.Embedding, essentially making the decoder layer
        # effectively do a "lookup" in the embedding features -- the decoder
        # will produce dot product similarity to 50K embeddings, and the
        # softmax/argmax after that will find the nearest embedding, and hence
        # the next token. In our case, since we produce a new feature which
        # may not be in the original set of features inputted (unlike a set
        # of exhaustive tokens), I would need a layer to generate the new
        # feature.
        self.decoder = nn.Linear(inter_dim, in_features, bias=False)
        # If encoder is an embedding, then tie up the weights
        if isinstance(self.encoder, nn.Embedding):
            self.decoder.weight = self.encoder.weight
        if freeze_encoder_decoder:
            self.encoder.weight.requires_grad = False
            self.decoder.weight.requires_grad = False
        # This already has the LayerNorm inside residual, as Naman suggested.
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(n_embd=inter_dim,
                                    vocab_size=in_features,
                                    use_cache=True,
                                    **kwargs))
        # Not needed, encoder will take care of it. Keeping it gives the error
        # that there are untrained params in the model. Earlier probably it
        # worked since I was using the LMHead model, which probably tied up
        # weights to the decoder layer, and hence trained it maybe
        del self.gpt_model.wte
        self.output_len = output_len
        self.output_len_eval = output_len_eval
        self.avg_last_n = avg_last_n
        self.inter_dim = inter_dim
        self.in_features = in_features
        if future_pred_loss is not None:
            self.future_pred_loss = hydra.utils.instantiate(future_pred_loss,
                                                            reduction='none')
        else:
            self.future_pred_loss = None
        self.return_past_too = return_past_too
        self.drop_last_n = drop_last_n
        # Set this, if want to quantize the prediction (using top-1) and
        # re-encode, as opposed to using the soft predicted feature
        self.quantize_before_rollout = quantize_before_rollout

    def forward(self, feats, target_shape):
        """
        Args:
            feats: tensor of shape (B, T, C)
            target_shape: shape of the output (B, T', n_output)
        """
        addl_endpoints = {}
        if feats.ndim == 2:
            # add back the temporal dimension, which was likely mean pooled
            feats = feats.unsqueeze(1)
        # Decide the output len based on the target_shape
        if len(target_shape) == 3:
            output_len = target_shape[1]
        elif self.training or self.output_len_eval < 0:
            # If training mode or output_len for eval has not been set
            output_len = self.output_len
        else:  # eval mode
            output_len = self.output_len_eval
        # Keep track
        full_inp_feats = feats
        if self.assign_to_centroids:
            # Unsqueeze only to be compatible with the 1 channel inputs -- that
            # will get squeezed out later
            feats = self.assigner(feats).unsqueeze(-1)
        # The time dimension in already in the middle -> B, T, C
        # That's what huggingface version needs:
        # (batch_size, sequence_length, hidden_size)
        if self.in_features == 1 or self.assign_to_centroids:
            # This is a quantized input, so cast it to long, and remove the
            # last singleton dimension
            assert feats.size(-1) == 1
            feats = feats.squeeze(-1).long()
        # Keep only the first N, this is used when the model is given
        # input more frames than it should be using for prediction. The other
        # future is used to incur loss during training, but shouldn't otherwise
        # be used, so dropping those features
        full_orig_feats = feats
        inp_feats = full_inp_feats
        if self.drop_last_n != 0:
            logging.warning('This should be used very carefully, ideally only '
                            'for debugging. The padding can lead to some '
                            'frames from the actual clip to leak into the '
                            'past clip, even after dropping last n. So even '
                            'after dropping the model might end up seeing '
                            'frames that are beyond the tau_a.')
            feats = feats[:, :-self.drop_last_n]
            inp_feats = inp_feats[:, :-self.drop_last_n]
        # Keep track
        orig_feats_len = feats.size(1)
        # Reduce the dimensionality, since not using the GPT encoding matrix,
        # since I don't have a "token" representation
        feats = self.encoder(feats)
        orig_feats_encoded = feats
        past = None
        all_outputs = []
        all_outputs_decoded = []
        for output_id in range(output_len):
            pred_so_far = sum([el.size(1) for el in all_outputs])
            position_ids = torch.arange(pred_so_far,
                                        pred_so_far + feats.size(1),
                                        dtype=torch.long,
                                        device=feats.device)
            # The past output will encode the previous past AND the new input
            # (you can check the output, it keeps increasing)
            # Got this from
            # https://huggingface.co/transformers/quickstart.html#using-the-past
            outputs = self.gpt_model(inputs_embeds=feats,
                                     past_key_values=past,
                                     position_ids=position_ids)
            last_hidden_state = outputs.last_hidden_state
            past = outputs.past_key_values
            all_outputs.append(last_hidden_state)
            # For visualization later, if output_attentions was passed into gpt
            if outputs.attentions is not None:
                # dimensions will be (batch_size, nlayers, nheads, seqlen, seqlen)
                addl_endpoints[f'gpt2_att_{output_id}'] = torch.stack(
                    outputs.attentions).transpose(0, 1)
            # Map back to the original feature dimension
            all_outputs_decoded.append(self.decoder(last_hidden_state))
            # hidden_states[-1] or last_hidden_state is the embedding from the
            # final layer. Not using logits (earlier was using the LMHead model
            # that returned logits) since that is already decoded to vocab size
            # and I want to have control over the weights of that final matrix
            # Also, the input for the next would be encodings, so need to
            # access the encodings directly
            if self.quantize_before_rollout:
                assert isinstance(self.encoder, nn.Embedding)
                feats = self.encoder(
                    all_outputs_decoded[-1][:, -1:, :].argmax(dim=-1))
            else:
                feats = last_hidden_state[:, -1:, :]
        all_outputs = torch.cat(all_outputs, dim=1)
        all_outputs_decoded = torch.cat(all_outputs_decoded, dim=1)
        # Compute a loss on future prediction (teacher forced)
        losses = {}
        if self.future_pred_loss is not None:
            num_elts_for_loss = min(full_orig_feats.size(1),
                                    all_outputs_decoded.size(1))
            losses = {
                'gpt2':
                self.future_pred_loss(
                    all_outputs_decoded[:, :num_elts_for_loss - 1],
                    full_orig_feats[:, 1:num_elts_for_loss])
            }
        # Set all_output as the final output features, and prev as the
        # structure to use to get the original features of past
        if self.in_features == 1:
            prev = orig_feats_encoded
            # all_outputs contains the hidden states, the best we will get
            # anyway, so that doesn't change
        elif self.assign_to_centroids:
            prev = inp_feats  # For this, I have the orig feats, so use that
            # For prediction, use the predicted cluster centers, but use
            # features from the original kmeans, not what the embeddings
            # that were learnt.. it didn't work with them
            all_outputs = self.assigner(all_outputs_decoded.argmax(dim=-1))
        else:
            prev = inp_feats
            all_outputs = all_outputs_decoded
        # Return the actual predictions
        if self.return_past_too:
            # Pad in the GT past (no point using the predicted past when
            # we have the actual past)
            final = torch.cat((prev, all_outputs[:, orig_feats_len - 1:, :]),
                              dim=1)
        elif output_len > 0:
            final = all_outputs[:, -output_len:]
        else:
            final = all_outputs
        if self.avg_last_n > 0:
            final = torch.mean(final[:, -self.avg_last_n:, :], dim=1)
        # compute the past feature.
        # Note all_outputs[:, :orig_feats_len] is NOT past -- since GPT
        # predicts the next feature, which is what we enforce in the GPT loss
        # as well (20210305)
        assert prev.size(1) == orig_feats_len, (
            'If not, need to figure how to deal')
        # Now keep the old feature for the first one, and return the predicted
        # features shifted by 1 for the rest -- which are as predicted by
        # GPT
        updated_past_feat = torch.cat(
            [prev[:, :1, :], all_outputs[:, :(orig_feats_len - 1)]], dim=1)
        return updated_past_feat, final, losses, addl_endpoints

    @property
    def output_dim(self):
        if self.in_features == 1:
            return self.inter_dim  # since it will return encoded features
        # else, it will decode it back to the original feat dimension
        return self.in_features
